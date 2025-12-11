import os 
import litellm
import torch
from typing import Optional
from config import TOGETHER_MODEL_NAMES, LITELLM_TEMPLATES, API_KEY_NAMES, Model, HF_MODEL_NAMES
from loggers import logger
from common import get_api_key

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = Model(model_name)
    
    def batched_generate(self, prompts_list: list, max_n_tokens: int, temperature: float):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
class APILiteLLM(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "ERROR: API CALL FAILED."
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 20

    def __init__(self, model_name):
        super().__init__(model_name)
        self.api_key = get_api_key(self.model_name)
        self.litellm_model_name = self.get_litellm_model_name(self.model_name)
        litellm.drop_params=True
        self.set_eos_tokens(self.model_name)
        
    def get_litellm_model_name(self, model_name):
        if model_name in TOGETHER_MODEL_NAMES:
            litellm_name = TOGETHER_MODEL_NAMES[model_name]
            self.use_open_source_model = True
        else:
            self.use_open_source_model =  False
            #if self.use_open_source_model:
                # Output warning, there should be a TogetherAI model name
                #logger.warning(f"Warning: No TogetherAI model name for {model_name}.")
            litellm_name = model_name.value 
        return litellm_name
    
    def set_eos_tokens(self, model_name):
        if self.use_open_source_model:
            self.eos_tokens = LITELLM_TEMPLATES[model_name]["eos_tokens"]     
        else:
            self.eos_tokens = []

    def _update_prompt_template(self):
        # We manually add the post_message later if we want to seed the model response
        if self.model_name in LITELLM_TEMPLATES:
            litellm.register_prompt_template(
                initial_prompt_value=LITELLM_TEMPLATES[self.model_name]["initial_prompt_value"],
                model=self.litellm_model_name,
                roles=LITELLM_TEMPLATES[self.model_name]["roles"]
            )
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.post_message = ""
        
    
    
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]: 
        
        eos_tokens = self.eos_tokens 

        if extra_eos_tokens:
            eos_tokens.extend(extra_eos_tokens)
        if self.use_open_source_model:
            self._update_prompt_template()
        
        outputs = litellm.batch_completion(
            model=self.litellm_model_name, 
            messages=convs_list,
            api_key=self.api_key,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_n_tokens,
            num_retries=self.API_MAX_RETRY,
            seed=0,
            stop=eos_tokens,
        )
        
        responses = [output["choices"][0]["message"].content for output in outputs]

        return responses


class LocalTransformers(LanguageModel):
    """Local model using HuggingFace transformers with optional PEFT adapter support."""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None, 
                 peft_adapter_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize local model with transformers.
        
        Args:
            model_name: Model identifier from config.Model enum
            model_path: Path to local model or HF model name. If None, uses path from config.
            peft_adapter_path: Optional path to PEFT adapter (LoRA, etc.)
            device: Device to load model on
        """
        super().__init__(model_name)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Determine model path
        if model_path is None:
            # Try to get from HF_MODEL_NAMES
            if self.model_name in HF_MODEL_NAMES:
                model_path = HF_MODEL_NAMES[self.model_name]
            else:
                raise ValueError(f"No model path specified and no default path for {model_name}")
        
        logger.info(f"Loading local model from {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        
        # Load PEFT adapter if specified
        if peft_adapter_path:
            logger.info(f"Loading PEFT adapter from {peft_adapter_path}")
            try:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, peft_adapter_path)
                logger.info("PEFT adapter loaded successfully")
            except ImportError:
                raise ImportError("PEFT library not installed. Install with: pip install peft")
        
        self.device = device
        if device != "cuda" and hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Get template info
        if self.model_name in LITELLM_TEMPLATES:
            self.eos_tokens = LITELLM_TEMPLATES[self.model_name]["eos_tokens"]
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.eos_tokens = []
            self.post_message = ""
            
        self.use_open_source_model = True
    
    def _format_messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert OpenAI-style messages to a prompt string using the model's chat template."""
        # Try to use the model's built-in chat template
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Failed to apply chat template: {e}. Falling back to manual formatting.")
        
        # Fallback: manual formatting using LITELLM_TEMPLATES
        if self.model_name not in LITELLM_TEMPLATES:
            # Simple fallback
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    prompt += f"System: {content}\n"
                elif role == "user":
                    prompt += f"User: {content}\n"
                elif role == "assistant":
                    prompt += f"Assistant: {content}\n"
            prompt += "Assistant:"
            return prompt
        
        # Use LITELLM template
        template = LITELLM_TEMPLATES[self.model_name]
        prompt = template.get("initial_prompt_value", "")
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if content:  # Only process non-empty messages
                role_template = template["roles"].get(role, template["roles"]["user"])
                prompt += role_template["pre_message"] + content + role_template["post_message"]
        
        return prompt
    
    def batched_generate(self, convs_list: list[list[dict]], 
                         max_n_tokens: int, 
                         temperature: float, 
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]:
        """
        Generate responses for a batch of conversations.
        
        Args:
            convs_list: List of conversation histories in OpenAI message format
            max_n_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            extra_eos_tokens: Additional tokens to stop generation
            
        Returns:
            List of generated responses
        """
        # Convert conversations to prompts
        prompts = [self._format_messages_to_prompt(conv) for conv in convs_list]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=2048  # Adjust based on model context length
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Prepare stop tokens
        stop_strings = self.eos_tokens.copy()
        if extra_eos_tokens:
            stop_strings.extend(extra_eos_tokens)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_n_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode outputs
        generated_texts = []
        for i, output in enumerate(outputs):
            # Remove the input prompt tokens
            input_length = inputs['input_ids'][i].shape[0]
            generated_ids = output[input_length:]
            
            # Decode
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Remove stop strings
            for stop_str in stop_strings:
                if stop_str in text:
                    text = text[:text.index(stop_str)]
            
            generated_texts.append(text.strip())
        
        return generated_texts


# class LocalvLLM(LanguageModel):
    
#     def __init__(self, model_name: str):
#         """Initializes the LLMHuggingFace with the specified model name."""
#         super().__init__(model_name)
#         if self.model_name not in MODEL_NAMES:
#             raise ValueError(f"Invalid model name: {model_name}")
#         self.hf_model_name = HF_MODEL_NAMES[Model(model_name)]
#         destroy_model_parallel()
#         self.model = vllm.LLM(model=self.hf_model_name)
#         if self.temperature > 0:
#             self.sampling_params = vllm.SamplingParams(
#                 temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_n_tokens
#             )
#         else:
#             self.sampling_params = vllm.SamplingParams(temperature=0, max_tokens=self.max_n_tokens)

#     def _get_responses(self, prompts_list: list[str], max_new_tokens: int | None = None) -> list[str]:
#         """Generates responses from the model for the given prompts."""
#         full_prompt_list = self._prompt_to_conv(prompts_list)
#         outputs = self.model.generate(full_prompt_list, self.sampling_params)
#         # Get output from each input, but remove initial space
#         outputs_list = [output.outputs[0].text[1:] for output in outputs]
#         return outputs_list

#     def _prompt_to_conv(self, prompts_list):
#         batchsize = len(prompts_list)
#         convs_list = [self._init_conv_template() for _ in range(batchsize)]
#         full_prompts = []
#         for conv, prompt in zip(convs_list, prompts_list):
#             conv.append_message(conv.roles[0], prompt)
#             conv.append_message(conv.roles[1], None)
#             full_prompt = conv.get_prompt()
#             # Need this to avoid extraneous space in generation
#             if "llama-2-7b-chat-hf" in self.model_name:
#                 full_prompt += " "
#             full_prompts.append(full_prompt)
#         return full_prompts

#     def _init_conv_template(self):
#         template = get_conversation_template(self.hf_model_name)
#         if "llama" in self.hf_model_name:
#             # Add the system prompt for Llama as FastChat does not include it
#             template.system_message = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
#         return template
    






