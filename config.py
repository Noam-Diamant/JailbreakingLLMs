import glob as _glob
import os as _os
from enum import Enum
VICUNA_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--lmsys--vicuna-13b-v1.5"
LLAMA_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--meta-llama--Llama-2-7b-chat-hf"
LLAMA_3_8B_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--meta-llama--Meta-Llama-3-8B"
LLAMA_3_1_8B_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
GEMMA_2_2B_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--google--gemma-2-2b"
QWEN_57B_GPTQ_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--Qwen--Qwen2-57B-A14B-Instruct-GPTQ-Int4"
LLAMA_GUARD_3_8B_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--meta-llama--Llama-Guard-3-8B"
ZEPHYR_7B_PATH = "/dsi/fetaya-lab/noam_diamant/hugging_face/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/892b3d7a7b1cf10c7a701c60881cd93df615734c"

ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1


## MODEL PARAMETERS ##
class Model(Enum):
    vicuna = "vicuna-13b-v1.5"
    llama_2 = "llama-2-7b-chat-hf"
    llama_3_8b = "meta-llama-3-8b"
    llama_3_1_8b = "llama-3.1-8b"
    gpt_3_5 = "gpt-3.5-turbo-1106"
    gpt_4 = "gpt-4-0125-preview"
    claude_1 = "claude-instant-1.2"
    claude_2 = "claude-2.1"
    gemini = "gemini-pro"
    mixtral = "mixtral"
    gemma_2_2b = "gemma-2-2b"
    qwen_57b_gptq = "qwen2-57b-a14b-instruct-gptq-int4"
    llama_guard_3_8b = "llama-guard-3-8b"
    zephyr_7b = "zephyr-7b"

MODEL_NAMES = [model.value for model in Model]


HF_MODEL_NAMES: dict[Model, str] = {
    Model.llama_2: "meta-llama/Llama-2-7b-chat-hf",
    Model.llama_3_8b: LLAMA_3_8B_PATH,
    Model.llama_3_1_8b: "meta-llama/Llama-3.1-8B",
    Model.vicuna: "lmsys/vicuna-13b-v1.5",
    Model.mixtral: "mistralai/Mixtral-8x7B-Instruct-v0.1",
    Model.gemma_2_2b: "google/gemma-2-2b",
    Model.qwen_57b_gptq: "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
    Model.llama_guard_3_8b: LLAMA_GUARD_3_8B_PATH,
    Model.zephyr_7b: ZEPHYR_7B_PATH,
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
    Model.llama_3_1_8b: "llama-3",
    Model.mixtral: "mixtral",
    Model.gemma_2_2b: "gemma",
    Model.qwen_57b_gptq: "qwen-7b-chat",
    Model.llama_guard_3_8b: "llama-3",
    # Zephyr: fastchat falls back to one_shot; actual formatting uses the
    # tokenizer's built-in apply_chat_template (Zephyr has one natively).
    Model.zephyr_7b: "zephyr-7b",
}

API_KEY_NAMES: dict[Model, str] = {
    Model.gpt_3_5:  "OPENAI_API_KEY",
    Model.gpt_4:    "OPENAI_API_KEY",
    Model.claude_1: "ANTHROPIC_API_KEY",
    Model.claude_2: "ANTHROPIC_API_KEY",
    Model.gemini:   "GEMINI_API_KEY",
    Model.vicuna:   "TOGETHER_API_KEY",
    Model.llama_2:  "TOGETHER_API_KEY",
    Model.llama_3_8b: "TOGETHER_API_KEY",
    Model.llama_3_1_8b: "TOGETHER_API_KEY",
    Model.mixtral:  "TOGETHER_API_KEY",
    Model.gemma_2_2b: "TOGETHER_API_KEY",
    Model.qwen_57b_gptq: "TOGETHER_API_KEY",
    Model.zephyr_7b: "TOGETHER_API_KEY",
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
    Model.llama_3_1_8b: {"roles":{
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
    },
    Model.llama_guard_3_8b: {"roles":{
                    "system": {"pre_message": "<|start_header_id|>system<|end_header_id|>\n\n", "post_message": "<|eot_id|>"},
                    "user": {"pre_message": "<|start_header_id|>user<|end_header_id|>\n\n", "post_message": "<|eot_id|>"},
                    "assistant": {"pre_message": "<|start_header_id|>assistant<|end_header_id|>\n\n", "post_message": "<|eot_id|>"},
                },
                "post_message" : "",
                "initial_prompt_value" : "<|begin_of_text|>",
                "eos_tokens" :  ["<|eot_id|>"]
    },
    # Zephyr 7B beta — <|system|>…</s><|user|>…</s><|assistant|>
    # This entry is only used as a fallback; the tokenizer's apply_chat_template
    # is preferred and handles Zephyr's format natively.
    Model.zephyr_7b: {"roles":{
                    "system": {"pre_message": "<|system|>\n", "post_message": "</s>\n"},
                    "user": {"pre_message": "<|user|>\n", "post_message": "</s>\n"},
                    "assistant": {"pre_message": "<|assistant|>\n", "post_message": "</s>\n"},
                },
                "post_message": "",
                "initial_prompt_value": "",
                "eos_tokens": ["</s>"]
    },
}


# =============================================================================
# Unlearned model registry
# =============================================================================
#
# Maps short string keys → model metadata dicts so callers never hard-code paths.
#
# Each entry has exactly two fields:
#   "base"  – Model enum value string (e.g. "zephyr-7b").  Determines which
#             architecture / tokenizer to use.  When the path is a PEFT adapter
#             the base model's weights (from HF_MODEL_NAMES) are loaded first.
#   "path"  – Absolute path to either a full model OR a PEFT adapter directory.
#             The distinction is detected automatically at call time by checking
#             for `adapter_config.json` inside the directory:
#               * present  → PEFT adapter (load base weights + apply adapter)
#               * absent   → full fine-tuned model (replaces base weights)
#
# Helper:  get_unlearned_model_args(key) → {base_model, model_path, peft_adapter}
#
# The helper translates registry entries into the three CLI-argument equivalents:
#   --target-model        ← base_model
#   --target-model-path   ← model_path   (full models only; None for PEFT)
#   --target-peft-adapter ← peft_adapter (PEFT only; None for full models)
# =============================================================================

_PROJECT = "/dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE"


def _p(*parts: str) -> str:
    """Join path parts under the project root."""
    return _os.path.join(_PROJECT, *parts)


def _latest_glob(pattern: str) -> str:
    """
    Expand a glob pattern and return the most recently modified match.
    Raises FileNotFoundError if nothing matches.
    """
    matches = _glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No directories matched the glob pattern: {pattern!r}"
        )
    return max(matches, key=_os.path.getmtime)


def _is_peft_adapter(path: str) -> bool:
    """
    Return True when *path* is a PEFT/LoRA adapter directory.

    Detection rule: HuggingFace PEFT always writes `adapter_config.json` to the
    adapter root when calling `save_pretrained`.  Full fine-tuned models never
    have this file.  This is the same heuristic used by `peft.PeftModel` itself
    when calling `from_pretrained`.
    """
    return _os.path.isfile(_os.path.join(path, "adapter_config.json"))


# Raw registry — just base model + path.  PEFT vs full is auto-detected.
_UNLEARNED_MODEL_REGISTRY: dict[str, dict] = {
    # ------------------------------------------------------------------
    # Base / original (unmodified) models
    # ------------------------------------------------------------------
    "zephyr_base": {
        "base": "zephyr-7b",
        "path": ZEPHYR_7B_PATH,
    },
    "llama3_base": {
        "base": "meta-llama-3-8b",
        "path": LLAMA_3_8B_PATH,
    },

    # ------------------------------------------------------------------
    # RMU — weights are fully replaced (no PEFT)
    # ------------------------------------------------------------------
    "zephyr_rmu": {
        "base": "zephyr-7b",
        "path": _p("wmdp/models/Zephyr_RMU_original_hugging_face"),
    },
    "llama3_rmu_bio": {
        "base": "meta-llama-3-8b",
        "path": _p("wmdp/models/llama3_rmu_bio_5_25"),
    },
    "llama3_rmu_cyber": {
        "base": "meta-llama-3-8b",
        "path": _p("wmdp/models/llama3_rmu_cyber_5_25"),
    },

    # ------------------------------------------------------------------
    # ELM — saves a LoRA adapter on top of the base model (PEFT)
    #
    # Note: elm-zephyr-7b-beta is the adapter for Zephyr,
    #       elm-Meta-Llama-3-8B is the adapter for Llama-3.
    # (The path names are the authoritative source of truth here.)
    # ------------------------------------------------------------------
    "zephyr_elm": {
        "base": "zephyr-7b",
        "path": _p("elm/models/elm-zephyr-7b-beta"),
    },
    "llama3_elm": {
        "base": "meta-llama-3-8b",
        "path": _p("elm/models/elm-Meta-Llama-3-8B"),
    },

    # ------------------------------------------------------------------
    # SNPO — full fine-tuned checkpoints
    # ------------------------------------------------------------------
    "zephyr_snpo": {
        "base": "zephyr-7b",
        "path": _p("snpo/WMDP/files/results/OPTML-Group_NPO-WMDP"),
    },
    "llama3_snpo": {
        "base": "meta-llama-3-8b",
        # Resolved lazily at call time via get_unlearned_model_args()
        "path_glob": _p(
            "snpo/WMDP/files/results/unlearn_wmdp_bio_cyber/"
            "NPO_llama3_8b/*/checkpoints"
        ),
    },

    # ------------------------------------------------------------------
    # SimNPO — full fine-tuned checkpoints
    # ------------------------------------------------------------------
    "zephyr_simnpo": {
        "base": "zephyr-7b",
        "path": _p("simnpo/WMDP/files/results/SimNPO_WMDP_zephyr_7b_beta"),
    },
    "llama3_simnpo": {
        "base": "meta-llama-3-8b",
        # Resolved lazily at call time via get_unlearned_model_args()
        "path_glob": _p(
            "simnpo/WMDP/files/results/unlearn_wmdp_bio_cyber/"
            "SimNPO_llama3_8b/*/checkpoints"
        ),
    },
}

UNLEARNED_MODEL_KEYS = list(_UNLEARNED_MODEL_REGISTRY.keys())


def get_unlearned_model_args(key: str) -> dict:
    """
    Return a dict ready to be applied to argparse args for the given key.

    Returns
    -------
    {
        "base_model":   str   – value for --target-model
        "model_path":   str | None – value for --target-model-path
                          (None when path is a PEFT adapter)
        "peft_adapter": str | None – value for --target-peft-adapter
                          (None when path is a full model)
        "is_peft":      bool  – True when a PEFT adapter was detected
    }

    PEFT detection
    --------------
    The function checks for `adapter_config.json` inside the resolved path.
    This file is always written by `peft.PeftModel.save_pretrained` and is
    never present in ordinary HuggingFace full-model directories, making it
    a reliable and zero-configuration signal.
    """
    if key not in _UNLEARNED_MODEL_REGISTRY:
        raise KeyError(
            f"Unknown unlearned model key {key!r}. "
            f"Available: {UNLEARNED_MODEL_KEYS}"
        )

    entry = _UNLEARNED_MODEL_REGISTRY[key]

    # Resolve glob-based paths (lazy, so we get the latest checkpoint)
    if "path_glob" in entry:
        path = _latest_glob(entry["path_glob"])
    else:
        path = entry["path"]

    is_peft = _is_peft_adapter(path)

    return {
        "base_model":   entry["base"],
        "model_path":   None if is_peft else path,
        "peft_adapter": path if is_peft else None,
        "is_peft":      is_peft,
    }