#!/bin/bash
# Command to run main.py with DSPy judge
# Prerequisites:
# 1. Start vLLM server for judge/attack model on port 8005 (see instructions above)
# 2. Make sure the compiled DSPy judge exists at saved_judges/dspy_judge_optimized.json

cd /dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/JailbreakingLLMs

python main.py \
  --csv-file data/bio_mcq_jailbreak.csv \
  --max-csv-rows 100 \
  --attack-model qwen2-57b-a14b-instruct-gptq-int4 \
  --target-model llama-3.1-8b \
  --judge-model dspy \
  --judge-dspy-path saved_judges/dspy_judge_optimized.json \
  --judge-dspy-model qwen2-57b-a14b-instruct-gptq-int4 \
  --judge-api-base http://localhost:8005/v1 \
  --evaluate-locally \
  --use-vllm \
  --target-model-path llama-3.1-8b \
  --target-peft-adapter /dsi/fetaya-lab/noam_diamant/projects/Unlearning_with_SAE/CRISP/CRISP/saved_models/crisp/llama_3_1/bio \
  --attack-gpu 1 \
  -v > outputs/peft_run_dspy_judge.txt 2>&1

