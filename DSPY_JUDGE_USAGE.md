# DSPy Judge Usage Guide

This guide explains how to use DSPy to optimize your judge system prompts and integrate the optimized judge into your jailbreaking experiments.

## Overview

DSPy (Declarative Self-improving Python) is a framework that optimizes prompts and language model programs. We use it to:
1. Optimize the judge's system prompt based on training examples
2. Save the optimized judge for reuse
3. Integrate it seamlessly with your existing codebase

## Step 1: Install Dependencies

First, install DSPy:

```bash
pip install dspy-ai
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Training Data

Convert your CSV file (`data/bio_judge_scoring.csv`) to DSPy Examples format:

```bash
python prepare_dspy_data.py \
    --csv-file data/bio_judge_scoring.csv \
    --output-json data/dspy_judge_examples.json \
    --train-split 0.8
```

This will create:
- `data/dspy_judge_examples_train.json` - Training examples (80%)
- `data/dspy_judge_examples_val.json` - Validation examples (20%)
- `data/dspy_judge_examples.json` - All examples combined

## Step 3: Compile/Optimize the Judge

Before compiling, make sure your vLLM server is running for the judge model. If you're using the same model as attack/target, it should already be running.

**Option A: Using HTTP vLLM Server (Recommended)**

If your vLLM server is running on a specific port (e.g., port 8006 for judge):

```bash
python compile_dspy_judge.py \
    --train-data data/dspy_judge_examples_train.json \
    --val-data data/dspy_judge_examples_val.json \
    --save-path saved_judges/dspy_judge_optimized \
    --model-name qwen2-57b-a14b-instruct-gptq-int4 \
    --api-base http://localhost:8006/v1 \
    --optimizer bootstrap \
    --num-threads 4
```

**Option B: Auto-detect from Running Servers**

If you're running the main script with `--use-vllm`, the API bases are automatically assigned. You can check which port your judge model uses and use that.

## Step 4: Use the Optimized Judge

Once compiled, use the DSPy judge in your main script:

```bash
python main.py \
    --csv-file data/bio_mcq_jailbreak.csv \
    --max-csv-rows 100 \
    --attack-model qwen2-57b-a14b-instruct-gptq-int4 \
    --target-model llama-3.1-8b \
    --judge-model dspy \
    --judge-dspy-path saved_judges/dspy_judge_optimized \
    --judge-dspy-model qwen2-57b-a14b-instruct-gptq-int4 \
    --evaluate-locally \
    --use-vllm \
    --target-model-path llama-3.1-8b \
    --target-peft-adapter /path/to/adapter \
    --attack-gpu 1 \
    -v > outputs/peft_run.txt 2>&1
```

## Key Parameters

### For `compile_dspy_judge.py`:
- `--train-data`: Path to training examples JSON
- `--val-data`: Path to validation examples JSON (optional)
- `--save-path`: Where to save the compiled judge
- `--model-name`: Model identifier (e.g., `qwen2-57b-a14b-instruct-gptq-int4`)
- `--api-base`: HTTP API base URL for vLLM server (e.g., `http://localhost:8006/v1`)
- `--optimizer`: Optimizer to use (`bootstrap`, `mipro`, or `bettertogether`)
- `--num-threads`: Number of threads for parallel optimization

### For `main.py`:
- `--judge-model dspy`: Use DSPy judge
- `--judge-dspy-path`: Path to compiled judge (required when using `dspy` judge)
- `--judge-dspy-model`: Model name for DSPy judge (defaults to `qwen2-57b-a14b-instruct-gptq-int4`)

## How It Works

1. **DSPyJudge Class**: Located in `judges.py`, this class:
   - Loads the compiled/optimized judge from disk
   - Sets up DSPy to use your vLLM server via HTTP
   - Scores responses using the optimized prompts

2. **Integration**: The DSPy judge integrates seamlessly with your existing code:
   - Uses the same `JudgeBase` interface
   - Works with vLLM HTTP servers
   - Supports the same scoring format (0 or 1)

3. **Automatic Port Detection**: When using `--evaluate-locally --use-vllm`, the code automatically:
   - Assigns ports 8004, 8005, 8006 to unique models
   - Reuses ports for duplicate models
   - Sets `judge_api_base` automatically

## Optimizers

- **bootstrap**: Fast, good for getting started (recommended)
- **mipro**: More thorough optimization, slower
- **bettertogether**: Combines multiple optimizers

## Troubleshooting

1. **"DSPy not installed"**: Run `pip install dspy-ai`

2. **"No module named 'compile_dspy_judge'"**: Make sure you're running from the `JailbreakingLLMs/` directory

3. **"DSPy judge requires --judge-api-base"**: 
   - Make sure vLLM server is running
   - Provide `--judge-api-base` explicitly, or
   - Use `--evaluate-locally --use-vllm` to auto-detect

4. **Port conflicts**: Check which ports your vLLM servers are using:
   ```bash
   # Check running vLLM servers
   ps aux | grep vllm
   # Or check ports
   netstat -tuln | grep 800
   ```

## Example Workflow

```bash
# 1. Prepare data
python prepare_dspy_data.py --csv-file data/bio_judge_scoring.csv

# 2. Start vLLM servers (if not already running)
# See start_vllm_servers.sh for details

# 3. Compile judge
python compile_dspy_judge.py \
    --train-data data/dspy_judge_examples_train.json \
    --val-data data/dspy_judge_examples_val.json \
    --save-path saved_judges/dspy_judge_optimized \
    --model-name qwen2-57b-a14b-instruct-gptq-int4 \
    --api-base http://localhost:8005/v1

# 4. Run experiments with optimized judge
python main.py \
    --csv-file data/bio_mcq_jailbreak.csv \
    --judge-model dspy \
    --judge-dspy-path saved_judges/dspy_judge_optimized \
    --judge-dspy-model qwen2-57b-a14b-instruct-gptq-int4 \
    --evaluate-locally \
    --use-vllm \
    # ... other args
```

## Notes

- The compiled judge is saved once and can be reused across multiple experiments
- DSPy optimizes the prompts/instructions, not the model weights
- The judge still uses your specified model (e.g., `qwen2-57b-a14b-instruct-gptq-int4`) for inference
- DSPy works with any OpenAI-compatible API, including vLLM HTTP servers

