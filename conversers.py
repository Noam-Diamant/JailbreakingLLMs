from common import get_api_key, conv_template, extract_json
from language_models import APILiteLLM, LocalTransformers, LocalvLLM, HTTPChatModel
from config import FASTCHAT_TEMPLATE_NAMES, Model

# Models supported by JailbreakBench
JAILBREAKBENCH_SUPPORTED_MODELS = [
    "vicuna-13b-v1.5",
    "llama-2-7b-chat-hf", 
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview"
]

def load_attack_and_target_models(args):
    # create attack model and target model
    use_vllm = getattr(args, 'use_vllm', False)
    
    attackLM = AttackLM(
        model_name=args.attack_model, 
        max_n_tokens=args.attack_max_n_tokens, 
        max_n_attack_attempts=args.max_n_attack_attempts, 
        category=args.category,
        evaluate_locally=args.evaluate_locally,
        model_path=getattr(args, 'attack_model_path', None),
        peft_adapter_path=getattr(args, 'attack_peft_adapter', None),
        use_vllm=use_vllm,
        gpu_memory_utilization=getattr(args, 'attack_gpu_memory_utilization', 0.45),
        gpu_devices=getattr(args, 'attack_gpu', '0'),
        max_model_len=getattr(args, 'attack_max_model_len', None),
        api_base=getattr(args, 'attack_api_base', None),
        api_key="EMPTY",
    )
    
    targetLM = TargetLM(
        model_name=args.target_model,
        category=args.category,
        max_n_tokens=args.target_max_n_tokens,
        evaluate_locally=args.evaluate_locally,
        phase=args.jailbreakbench_phase,
        use_jailbreakbench=args.use_jailbreakbench,
        model_path=getattr(args, 'target_model_path', None),
        peft_adapter_path=getattr(args, 'target_peft_adapter', None),
        use_vllm=use_vllm,
        gpu_memory_utilization=getattr(args, 'target_gpu_memory_utilization', 0.45),
        gpu_devices=getattr(args, 'target_gpu', '0'),
        max_model_len=getattr(args, 'target_max_model_len', 8192),  # Default to 8192 to avoid KV cache issues
        api_base=getattr(args, 'target_api_base', None),
        api_key="EMPTY",
    )
    
    return attackLM, targetLM

def load_indiv_model(model_name, local = False, use_jailbreakbench=True, 
                     model_path=None, peft_adapter_path=None, use_vllm=False, 
                     gpu_memory_utilization=0.9, gpu_devices="0",
                     max_model_len=None,
                     api_base: str | None = None, api_key: str | None = None):
    """
    Load a model either via API or locally.
    
    Args:
        model_name: Model identifier
        local: Whether to load locally
        use_jailbreakbench: Whether to use JailbreakBench wrapper
        model_path: Optional custom path to model (for local loading)
        peft_adapter_path: Optional path to PEFT adapter (LoRA, etc.)
        use_vllm: Whether to use vLLM backend (faster) or HuggingFace Transformers
        gpu_memory_utilization: GPU memory utilization for vLLM (0.0 to 1.0)
        gpu_devices: GPU device(s) to use (e.g., "0" or "0,1,2,3")
    """
    # Highest priority: if an explicit HTTP API base is provided, use that.
    if api_base is not None:
        # #region agent log
        import json
        import time
        with open('/dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "pre-fix", "hypothesisId": "A", "location": "conversers.py:68", "message": "Using HTTPChatModel (api_base set)", "data": {"model_name": model_name, "api_base": api_base, "peft_adapter_path": peft_adapter_path, "local": local}, "timestamp": int(time.time() * 1000)}) + '\n')
        # #endregion
        return HTTPChatModel(
            model_name=model_name,
            api_base=api_base,
            api_key=api_key or "EMPTY",
        )

    if use_jailbreakbench: 
        if local:
            from jailbreakbench import LLMvLLM
            lm = LLMvLLM(model_name=model_name)
        else:
            from jailbreakbench import LLMLiteLLM
            api_key = get_api_key(Model(model_name))
            lm = LLMLiteLLM(model_name= model_name, api_key = api_key)
    else:
        if local:
            # Choose backend based on use_vllm flag
            if use_vllm:
                # #region agent log
                import json
                import time
                with open('/dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({"sessionId": "debug-session", "runId": "pre-fix", "hypothesisId": "A", "location": "conversers.py:87", "message": "Using LocalvLLM (local=True, use_vllm=True)", "data": {"model_name": model_name, "peft_adapter_path": peft_adapter_path, "model_path": model_path}, "timestamp": int(time.time() * 1000)}) + '\n')
                # #endregion
                # Use vLLM with optional LoRA adapter
                lm = LocalvLLM(
                    model_name=model_name,
                    model_path=model_path,
                    peft_adapter_path=peft_adapter_path,
                    gpu_memory_utilization=gpu_memory_utilization,
                    gpu_devices=gpu_devices,
                    max_model_len=max_model_len
                )
            else:
                # Use HuggingFace Transformers with optional PEFT adapter
                lm = LocalTransformers(
                    model_name=model_name,
                    model_path=model_path,
                    peft_adapter_path=peft_adapter_path
                )
        else:
            lm = APILiteLLM(model_name)
    return lm

class AttackLM():
    """
        Base class for attacker language models.
        
        Generates attacks for conversations using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
                model_name: str, 
                max_n_tokens: int, 
                max_n_attack_attempts: int, 
                category: str,
                evaluate_locally: bool,
                model_path: str = None,
                peft_adapter_path: str = None,
                use_vllm: bool = False,
                gpu_memory_utilization: float = 0.9,
                gpu_devices: str = "0",
                max_model_len: int | None = None,
                api_base: str | None = None,
                api_key: str | None = None):
        
        self.model_name = Model(model_name)
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts

        from config import ATTACK_TEMP, ATTACK_TOP_P
        self.temperature = ATTACK_TEMP
        self.top_p = ATTACK_TOP_P

        self.category = category
        self.evaluate_locally = evaluate_locally
        self.model = load_indiv_model(model_name, 
                                      local = evaluate_locally, 
                                      use_jailbreakbench=False,  # Cannot use JBB as attacker
                                      model_path=model_path,
                                      peft_adapter_path=peft_adapter_path,
                                      use_vllm=use_vllm,
                                      gpu_memory_utilization=gpu_memory_utilization,
                                      gpu_devices=gpu_devices,
                                      max_model_len=max_model_len,
                                      api_base=api_base,
                                      api_key=api_key,
                                      )

        # Only models that expose `use_open_source_model` and `post_message`
        # (APILiteLLM / LocalTransformers / LocalvLLM) support the JSON-seeding
        # trick for attacks. HTTPChatModel and other adapters simply skip it.
        self.initialize_output = (
            hasattr(self.model, "use_open_source_model")
            and getattr(self.model, "use_open_source_model")
        )
        self.template = FASTCHAT_TEMPLATE_NAMES[self.model_name]

    def preprocess_conversation(self, convs_list: list, prompts_list: list[str]):
        # For open source models, we can seed the generation with proper JSON
        init_message = ""
        if self.initialize_output:
            
            # Initalize the attack model's generated output to match format
            if len(convs_list[0].messages) == 0:# If is first message, don't need improvement
                init_message = '{"improvement": "","prompt": "'
            else:
                init_message = '{"improvement": "'
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if self.initialize_output:
                conv.append_message(conv.roles[1], init_message)
        openai_convs_list = [conv.to_openai_api_messages() for conv in convs_list]
        return openai_convs_list, init_message
        
    def _generate_attack(self, openai_conv_list: list[list[dict]], init_message: str):
        batchsize = len(openai_conv_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize
        new_adv_prompts = [None] * batchsize
        
        # Continuously generate outputs until all are valid or max_n_attack_attempts is reached
        for attempt in range(self.max_n_attack_attempts):
            # Subset conversations based on indices to regenerate
            convs_subset = [openai_conv_list[i] for i in indices_to_regenerate]
            # Generate outputs 
            outputs_list = self.model.batched_generate(convs_subset,
                                                        max_n_tokens = self.max_n_tokens,  
                                                        temperature = self.temperature,
                                                        top_p = self.top_p,
                                                        extra_eos_tokens=["}"]
                                                    )
            
            # Check for valid outputs and update the list
            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                # Check if model generated complete JSON (starts with '{') or just the continuation
                if full_output.strip().startswith('{'):
                    # Model generated complete JSON, just add closing brace
                    full_output = full_output + "}"
                else:
                    # Model generated continuation, prepend init_message
                    full_output = init_message + full_output + "}"
                attack_dict, json_str = extract_json(full_output)
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    new_adv_prompts[orig_index] = json_str
                else:
                    new_indices_to_regenerate.append(orig_index)
            
            # Update indices to regenerate for the next iteration
            indices_to_regenerate = new_indices_to_regenerate
            # If all outputs are valid, break
            if not indices_to_regenerate:
                break

        if any([output is None for output in valid_outputs]):
            raise ValueError(f"Failed to generate valid output after {self.max_n_attack_attempts} attempts. Terminating.")
        return valid_outputs, new_adv_prompts

    def get_attack(self, convs_list, prompts_list):
        """
        Generates responses for a batch of conversations and prompts using a language model. 
        Only valid outputs in proper JSON format are returned. If an output isn't generated 
        successfully after max_n_attack_attempts, it's returned as None.
        
        Parameters:
        - convs_list: List of conversation objects.
        - prompts_list: List of prompts corresponding to each conversation.
        
        Returns:
        - List of generated outputs (dictionaries) or None for failed generations.
        """
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        
        # Convert conv_list to openai format and add the initial message
        processed_convs_list, init_message = self.preprocess_conversation(convs_list, prompts_list)
        valid_outputs, new_adv_prompts = self._generate_attack(processed_convs_list, init_message)

        for jailbreak_prompt, conv in zip(new_adv_prompts, convs_list):
            # For open source models, we can seed the generation with proper JSON and omit the post message
            # We add it back here
            if self.initialize_output:
                jailbreak_prompt += self.model.post_message
            conv.update_last_message(jailbreak_prompt)
        
        return valid_outputs

class TargetLM():
    """
        Target language model for jailbreaking evaluation.
        Supports API-based, local, and PEFT fine-tuned models.
    """
    def __init__(self, 
            model_name: str, 
            category: str,
            max_n_tokens : int,
            phase: str,
            evaluate_locally: bool = False,
            use_jailbreakbench: bool = True,
            model_path: str = None,
            peft_adapter_path: str = None,
            use_vllm: bool = False,
            gpu_memory_utilization: float = 0.9,
            gpu_devices: str = "0",
            max_model_len: int | None = None,
            api_base: str | None = None,
            api_key: str | None = None):
        
        self.model_name = model_name
        self.max_n_tokens = max_n_tokens
        self.phase = phase
        
        # Automatically disable JailbreakBench for unsupported models
        if use_jailbreakbench and model_name not in JAILBREAKBENCH_SUPPORTED_MODELS:
            print(f"Warning: {model_name} is not supported by JailbreakBench. Using direct interface instead.")
            use_jailbreakbench = False
        
        # Disable JailbreakBench when using vLLM (jailbreakbench's vLLM support is just a dummy stub)
        if use_jailbreakbench and use_vllm:
            print(f"Warning: JailbreakBench does not support vLLM backend. Using direct vLLM interface instead.")
            use_jailbreakbench = False
        
        self.use_jailbreakbench = use_jailbreakbench
        self.evaluate_locally = evaluate_locally

        from config import TARGET_TEMP,  TARGET_TOP_P   
        self.temperature = TARGET_TEMP
        self.top_p = TARGET_TOP_P

        self.model = load_indiv_model(model_name, evaluate_locally, use_jailbreakbench,
                                      model_path=model_path,
                                      peft_adapter_path=peft_adapter_path,
                                      use_vllm=use_vllm,
                                      gpu_memory_utilization=gpu_memory_utilization,
                                      gpu_devices=gpu_devices,
                                      max_model_len=max_model_len,
                                      api_base=api_base,
                                      api_key=api_key)            
        
        # For non-JailbreakBench models, we need the template
        if not self.use_jailbreakbench:
            self.template = FASTCHAT_TEMPLATE_NAMES[Model(model_name)]
            
        self.category = category

    # ------------------------------------------------------------------
    # MCQ logit evaluation (no chat template, no generation)
    # ------------------------------------------------------------------

    def get_mcq_abcd_logits(self, raw_prompts: list) -> list:
        """
        Given raw prompt strings (adversarial prefix + MCQ block, no chat
        template wrapping), return A/B/C/D logit or log-probability values at
        the last token position for each prompt.

        Returns a list of dicts {'A': float, 'B': float, 'C': float, 'D': float}.

        - LocalTransformers: returns raw logits from model(**inputs).logits[:, -1, choice_idxs].
        - LocalvLLM: appends each letter to the prompt and reads P(letter|context)
          via prompt_logprobs=1 — no top-k approximation needed.
        - HTTPChatModel: not supported (raises NotImplementedError).
        """
        if isinstance(self.model, LocalvLLM):
            return self._mcq_logits_vllm(raw_prompts)
        elif isinstance(self.model, LocalTransformers):
            return self._mcq_logits_hf(raw_prompts)
        else:
            raise NotImplementedError(
                "MCQ logit scoring requires a local model (--evaluate-locally). "
                "HTTP-based target models (--target-api-base) are not supported "
                "for the 'mcq-logits' judge."
            )

    def _mcq_logits_hf(self, raw_prompts: list) -> list:
        """Forward pass via HuggingFace transformers; returns raw logits."""
        import torch

        m = self.model  # LocalTransformers instance
        tok = m.tokenizer

        A_id = tok.encode("A", add_special_tokens=False)[-1]
        B_id = tok.encode("B", add_special_tokens=False)[-1]
        C_id = tok.encode("C", add_special_tokens=False)[-1]
        D_id = tok.encode("D", add_special_tokens=False)[-1]
        choice_idxs = torch.tensor([A_id, B_id, C_id, D_id])

        orig_padding_side = tok.padding_side
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        device = next(m.model.parameters()).device
        inputs = tok(
            raw_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            logits_all = m.model(**inputs).logits  # (batch, seq, vocab)

        tok.padding_side = orig_padding_side

        # Last token of each (unpadded) sequence
        choice_logits = logits_all[:, -1, choice_idxs]  # (batch, 4)
        results = []
        for row in choice_logits:
            results.append({
                "A": row[0].item(),
                "B": row[1].item(),
                "C": row[2].item(),
                "D": row[3].item(),
            })
        return results

    def _mcq_logits_vllm(self, raw_prompts: list) -> list:
        """
        Forward pass via vLLM; returns log-probabilities for A/B/C/D.

        For each base prompt (ending with "Answer:\\n"), four variants are
        built by appending each letter ("A", "B", "C", "D").  vLLM's
        prompt_logprobs=1 is used to read P(letter | context) directly from
        the last prompt-token position, without relying on top-k sampling.

        This mirrors the HF path: we score exactly the four candidate tokens.
        """
        import vllm

        m = self.model  # LocalvLLM instance
        tok = m.tokenizer

        letters = ["A", "B", "C", "D"]
        letter_ids = [tok.encode(l, add_special_tokens=False)[-1] for l in letters]

        # Build 4×n_streams prompts, one per (base_prompt, letter) pair
        extended_prompts = [p + letter for p in raw_prompts for letter in letters]

        # prompt_logprobs=1 returns the conditional log-probability of every
        # input token given its left context; we only read the last position.
        sampling_params = vllm.SamplingParams(
            max_tokens=1,
            prompt_logprobs=1,
            temperature=0.0,
        )

        if m.peft_adapter_path:
            lora_req = vllm.lora.request.LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_local_path=m.peft_adapter_path,
            )
            outputs = m.model.generate(extended_prompts, sampling_params, lora_request=lora_req)
        else:
            outputs = m.model.generate(extended_prompts, sampling_params)

        results = []
        for base_idx in range(len(raw_prompts)):
            row = {}
            for letter_offset, letter in enumerate(letters):
                out = outputs[base_idx * 4 + letter_offset]
                # prompt_logprobs is a list of dicts (one per input token).
                # The last entry holds P(letter | context), keyed by token id.
                last_pos = out.prompt_logprobs[-1]  # {token_id: Logprob, ...}
                tid = letter_ids[letter_offset]
                if last_pos and tid in last_pos:
                    row[letter] = last_pos[tid].logprob
                else:
                    row[letter] = -100.0
            results.append(row)
        return results

    def get_response(self, prompts_list):
        if self.use_jailbreakbench:
            llm_response = self.model.query(prompts = prompts_list, 
                                behavior = self.category, 
                                phase = self.phase,
                                max_new_tokens=self.max_n_tokens)
            responses = llm_response.responses
        else:
            batchsize = len(prompts_list)
            convs_list = [conv_template(self.template) for _ in range(batchsize)]
            full_prompts = []
            for conv, prompt in zip(convs_list, prompts_list):
                conv.append_message(conv.roles[0], prompt)
                full_prompts.append(conv.to_openai_api_messages())

            responses = self.model.batched_generate(full_prompts, 
                                                            max_n_tokens = self.max_n_tokens,  
                                                            temperature = self.temperature,
                                                            top_p = self.top_p
                                                        )
           
        return responses