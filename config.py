from enum import Enum
VICUNA_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--lmsys--vicuna-13b-v1.5"
LLAMA_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--meta-llama--Llama-2-7b-chat-hf"
LLAMA_3_8B_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--meta-llama--Meta-Llama-3-8B"
GEMMA_2_2B_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--google--gemma-2-2b"
QWEN_57B_GPTQ_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--Qwen--Qwen2-57B-A14B-Instruct-GPTQ-Int4"

ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1


## MODEL PARAMETERS ##
class Model(Enum):
    vicuna = "vicuna-13b-v1.5"
    llama_2 = "llama-2-7b-chat-hf"
    llama_3_8b = "meta-llama-3-8b"
    gpt_3_5 = "gpt-3.5-turbo-1106"
    gpt_4 = "gpt-4-0125-preview"
    claude_1 = "claude-instant-1.2"
    claude_2 = "claude-2.1"
    gemini = "gemini-pro"
    mixtral = "mixtral"
    gemma_2_2b = "gemma-2-2b"
    qwen_57b_gptq = "qwen2-57b-a14b-instruct-gptq-int4"

MODEL_NAMES = [model.value for model in Model]


HF_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "meta-llama/Llama-2-7b-chat-hf",
    Model.llama_3_8b: "meta-llama/Meta-Llama-3-8B",
    Model.vicuna: "lmsys/vicuna-13b-v1.5",
    Model.mixtral: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    Model.gemma_2_2b: "google/gemma-2-2b",
    Model.qwen_57b_gptq: "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4"
}

TOGETHER_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "together_ai/togethercomputer/llama-2-7b-chat",
    Model.vicuna: "together_ai/lmsys/vicuna-13b-v1.5",
    Model.mixtral: "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    Model.gemma_2_2b: "together_ai/google/gemma-2-2b-it",
    Model.qwen_57b_gptq: "together_ai/Qwen/Qwen2-57B-A14B-Instruct"
}

FASTCHAT_TEMPLATE_NAMES: dict[Model, str] = {
    Model.gpt_3_5: "gpt-3.5-turbo",
    Model.gpt_4: "gpt-4",
    Model.claude_1: "claude-instant-1.2",
    Model.claude_2: "claude-2.1",
    Model.gemini: "gemini-pro",
    Model.vicuna: "vicuna_v1.1",
    Model.llama_2: "llama-2-7b-chat-hf",
    Model.llama_3_8b: "llama-3",
    Model.mixtral: "mixtral",
    Model.gemma_2_2b: "gemma",
    Model.qwen_57b_gptq: "qwen-7b-chat",
}

API_KEY_NAMES: dict[Model, str] = {
    Model.gpt_3_5:  "OPENAI_API_KEY",
    Model.gpt_4:    "OPENAI_API_KEY",
    Model.claude_1: "ANTHROPIC_API_KEY",
    Model.claude_2: "ANTHROPIC_API_KEY",
    Model.gemini:   "GEMINI_API_KEY",
    Model.vicuna:   "TOGETHER_API_KEY",
    Model.llama_2:  "TOGETHER_API_KEY",
    Model.mixtral:  "TOGETHER_API_KEY",
    Model.gemma_2_2b: "TOGETHER_API_KEY",
    Model.qwen_57b_gptq: "TOGETHER_API_KEY",
}

LITELLM_TEMPLATES: dict[Model, dict] = {
    Model.vicuna: {"roles":{
                    "system": {"pre_message": "", "post_message": " "},
                    "user": {"pre_message": "USER: ", "post_message": " ASSISTANT:"},
                    "assistant": {
                        "pre_message": "",
                        "post_message": "",
                    },
                },
                "post_message":"</s>",
                "initial_prompt_value" : "",
                "eos_tokens": ["</s>"]
                },
    Model.llama_2: {"roles":{
                    "system": {"pre_message": "[INST] <<SYS>>\n", "post_message": "\n<</SYS>>\n\n"},
                    "user": {"pre_message": "", "post_message": " [/INST]"},
                    "assistant": {"pre_message": "", "post_message": ""},
                },
                "post_message" : " </s><s>",
                "initial_prompt_value" : "",
                "eos_tokens" :  ["</s>", "[/INST]"]
            },
    Model.llama_3_8b: {"roles":{
                    "system": {"pre_message": "<|start_header_id|>system<|end_header_id|>\n\n", "post_message": "<|eot_id|>"},
                    "user": {"pre_message": "<|start_header_id|>user<|end_header_id|>\n\n", "post_message": "<|eot_id|>"},
                    "assistant": {"pre_message": "<|start_header_id|>assistant<|end_header_id|>\n\n", "post_message": "<|eot_id|>"},
                },
                "post_message" : "",
                "initial_prompt_value" : "<|begin_of_text|>",
                "eos_tokens" :  ["<|eot_id|>"]
            },
    Model.mixtral: {"roles":{
                    "system": {
                        "pre_message": "[INST] ",
                        "post_message": " [/INST]"
                    },
                    "user": { 
                        "pre_message": "[INST] ",
                        "post_message": " [/INST]"
                    }, 
                    "assistant": {
                        "pre_message": " ",
                        "post_message": "",
                    }
                },
                "post_message": "</s>",
                "initial_prompt_value" : "<s>",
                "eos_tokens": ["</s>", "[/INST]"]
    },
    Model.gemma_2_2b: {"roles":{
                    "system": {
                        "pre_message": "<start_of_turn>user\n",
                        "post_message": "<end_of_turn>\n"
                    },
                    "user": {
                        "pre_message": "<start_of_turn>user\n",
                        "post_message": "<end_of_turn>\n"
                    },
                    "assistant": {
                        "pre_message": "<start_of_turn>model\n",
                        "post_message": "<end_of_turn>\n"
                    }
                },
                "post_message": "",
                "initial_prompt_value": "<bos>",
                "eos_tokens": ["<end_of_turn>", "<eos>"]
    },
    Model.qwen_57b_gptq: {"roles":{
                    "system": {
                        "pre_message": "<|im_start|>system\n",
                        "post_message": "<|im_end|>\n"
                    },
                    "user": {
                        "pre_message": "<|im_start|>user\n",
                        "post_message": "<|im_end|>\n"
                    },
                    "assistant": {
                        "pre_message": "<|im_start|>assistant\n",
                        "post_message": "<|im_end|>\n"
                    }
                },
                "post_message": "",
                "initial_prompt_value": "",
                "eos_tokens": ["<|im_end|>", "<|endoftext|>"]
    }
}