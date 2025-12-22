import os 
import asyncio
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


class HTTPChatModel(LanguageModel):
    """
    Language model that talks to an existing OpenAI-compatible HTTP server
    (e.g., vLLM started with `vllm serve ... --api-key EMPTY`).

    This is used when we want `JailbreakingLLMs` to use already-running vLLM
    servers instead of starting models in-process.
    """

    def __init__(self, model_name: str, api_base: str, api_key: str = "EMPTY"):
        super().__init__(model_name)
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for HTTPChatModel. "
                "Install with: pip install openai"
            ) from e

        self.api_base = api_base
        self.api_key = api_key
        self.AsyncOpenAI = AsyncOpenAI

        # HTTP-backed models are treated as non-open-source for the purposes of
        # AttackLM's JSON-seeding logic. We don't prepend any structured JSON
        # template, and there is no post_message suffix.
        self.use_open_source_model = False
        self.post_message = ""

        # vLLM exposes the underlying HF repo id as the model name
        if self.model_name in HF_MODEL_NAMES:
            self.served_model_name = HF_MODEL_NAMES[self.model_name]
        else:
            # fall back to raw value (e.g., gpt-3.5-turbo-1106)
            self.served_model_name = self.model_name.value

    async def _aget_single(self, client, messages, max_n_tokens, temperature, top_p):
        response = await client.chat.completions.create(
            model=self.served_model_name,
            messages=messages,
            max_tokens=max_n_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content

    async def _aget_batch(self, convs_list, max_n_tokens, temperature, top_p):
        client = self.AsyncOpenAI(base_url=self.api_base, api_key=self.api_key)
        tasks = [
            self._aget_single(client, messages, max_n_tokens, temperature, top_p)
            for messages in convs_list
        ]
        return await asyncio.gather(*tasks)

    def batched_generate(self, convs_list: list[list[dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float,
                         extra_eos_tokens: list[str] = None) -> list[str]:
        """
        Synchronously generate a batch of responses by internally using
        an async OpenAI client against the running vLLM HTTP server.
        """
        # We ignore extra_eos_tokens here; stopping is handled by the server.
        return asyncio.run(
            self._aget_batch(convs_list, max_n_tokens, temperature, top_p)
        )
    
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


class LocalvLLM(LanguageModel):
    """Local model using vLLM with optional LoRA adapter support."""
    
    def __init__(self, model_name: str, model_path: Optional[str] = None,
                 peft_adapter_path: Optional[str] = None, 
                 gpu_memory_utilization: float = 0.9,
                 max_model_len: Optional[int] = None,
                 gpu_devices: str = "0"):
        """
        Initialize vLLM model with optional LoRA adapter.
        
        Args:
            model_name: Model identifier from config.Model enum
            model_path: Path to local model or HF model name. If None, uses path from config.
            peft_adapter_path: Optional path to LoRA adapter
            gpu_memory_utilization: GPU memory fraction to use (default 0.9)
            max_model_len: Maximum sequence length (default: model's default)
            gpu_devices: GPU device(s) to use (e.g., "0" or "0,1,2,3"). Set via CUDA_VISIBLE_DEVICES.
        """
        super().__init__(model_name)
        
        try:
            import vllm
        except ImportError:
            raise ImportError("vLLM not installed. Install with: pip install vllm")
        
        # Set CUDA_VISIBLE_DEVICES for this model
        # This limits vLLM to only see the specified GPU(s)
        original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_devices
        logger.info(f"Setting CUDA_VISIBLE_DEVICES={gpu_devices} for {model_name}")
        
        # Determine model path
        if model_path is None:
            if self.model_name in HF_MODEL_NAMES:
                model_path = HF_MODEL_NAMES[self.model_name]
            else:
                raise ValueError(f"No model path specified and no default path for {model_name}")
        
        logger.info(f"Loading vLLM model from {model_path}")
        
        # Configure vLLM
        vllm_kwargs = {
            "model": model_path,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
        }
        
        if max_model_len:
            vllm_kwargs["max_model_len"] = max_model_len
        
        # Add LoRA adapter if specified
        if peft_adapter_path:
            logger.info(f"Loading LoRA adapter from {peft_adapter_path}")
            vllm_kwargs["enable_lora"] = True
            vllm_kwargs["max_lora_rank"] = 64  # Adjust based on your adapter
        
        # Initialize vLLM
        self.model = vllm.LLM(**vllm_kwargs)
        self.peft_adapter_path = peft_adapter_path
        
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_devices is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            del os.environ['CUDA_VISIBLE_DEVICES']
        
        # Get template info
        if self.model_name in LITELLM_TEMPLATES:
            self.eos_tokens = LITELLM_TEMPLATES[self.model_name]["eos_tokens"]
            self.post_message = LITELLM_TEMPLATES[self.model_name]["post_message"]
        else:
            self.eos_tokens = []
            self.post_message = ""
        
        self.use_open_source_model = True
        
        # Get tokenizer for chat template formatting
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}. Will use manual formatting.")
            self.tokenizer = None
    
    def _format_messages_to_prompt(self, messages: list[dict]) -> str:
        """Convert OpenAI-style messages to a prompt string using the model's chat template."""
        # Try to use the tokenizer's built-in chat template
        if self.tokenizer and hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
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
        Generate responses for a batch of conversations using vLLM.
        
        Args:
            convs_list: List of conversation histories in OpenAI message format
            max_n_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            extra_eos_tokens: Additional tokens to stop generation
            
        Returns:
            List of generated responses
        """
        import vllm
        
        # Convert conversations to prompts
        prompts = [self._format_messages_to_prompt(conv) for conv in convs_list]
        
        # Prepare stop tokens
        stop_tokens = self.eos_tokens.copy() if self.eos_tokens else []
        if extra_eos_tokens:
            stop_tokens.extend(extra_eos_tokens)
        
        # Configure sampling parameters
        sampling_params = vllm.SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=top_p,
            max_tokens=max_n_tokens,
            stop=stop_tokens if stop_tokens else None,
        )
        
        # Generate with LoRA if specified
        if self.peft_adapter_path:
            # Create LoRA request for each prompt
            lora_requests = [
                vllm.lora.request.LoRARequest(
                    lora_name="adapter",
                    lora_int_id=1,
                    lora_local_path=self.peft_adapter_path
                )
                for _ in prompts
            ]
            outputs = self.model.generate(prompts, sampling_params, lora_request=lora_requests[0])
        else:
            outputs = self.model.generate(prompts, sampling_params)
        
        # Extract generated text
        generated_texts = []
        for output in outputs:
            text = output.outputs[0].text
            # Clean up any remaining stop tokens
            for stop_token in stop_tokens:
                if stop_token in text:
                    text = text[:text.index(stop_token)]
            generated_texts.append(text.strip())
        
        return generated_texts
    






