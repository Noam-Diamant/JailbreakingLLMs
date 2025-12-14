# GPU Assignment Guide

This guide explains how to control GPU assignment for attack, target, and judge models in JailbreakingLLMs.

## Overview

You can now run each model (attack, target, judge) on any GPU configuration:
- All models on the same GPU
- Each model on a separate GPU
- Any combination (e.g., attack+target on GPU 0, judge on GPU 1)
- Multi-GPU for single models (e.g., `0,1,2,3` for tensor/pipeline parallelism)

## New CLI Arguments

### GPU Assignment
- `--attack-gpu DEVICE` - GPU device(s) for attack model (default: `0`)
- `--target-gpu DEVICE` - GPU device(s) for target model (default: `0`)
- `--judge-gpu DEVICE` - GPU device(s) for judge model (default: `0`)

**Format:** Single GPU `0` or multiple GPUs `0,1,2,3`

### Judge Local Execution
- `--evaluate-judge-locally` - Run judge locally with vLLM (instead of API)
- `--judge-model-path PATH` - Path to local judge model
- `--judge-peft-adapter PATH` - Path to PEFT adapter for judge
- `--judge-gpu-memory-utilization FLOAT` - GPU memory % for judge (default: `0.45`)

## Memory Utilization

Default GPU memory utilization is **0.45 (45%)** per model to avoid OOM when multiple models share a GPU.

**Available arguments:**
- `--attack-gpu-memory-utilization FLOAT` (default: 0.45)
- `--target-gpu-memory-utilization FLOAT` (default: 0.45)
- `--judge-gpu-memory-utilization FLOAT` (default: 0.45)

### Calculating Memory Usage

On a GPU with **X GB** total memory:
- `0.45` utilization = 45% of X GB per model
- Multiple models on same GPU: sum their allocations

**Example with 80GB GPU:**
- 3 models at 0.45 each = 3 × 36GB = 108GB ❌ **OOM!**
- 3 models at 0.30 each = 3 × 24GB = 72GB ✅ **OK**
- 2 models at 0.45 each = 2 × 36GB = 72GB ✅ **OK**

## Usage Examples

### Example 1: All on same GPU (default)
```bash
python main.py \
  --evaluate-locally \
  --use-vllm \
  --attack-model vicuna-13b-v1.5 \
  --target-model gemma-2-2b \
  --judge-model no-judge \
  --attack-gpu 0 \
  --target-gpu 0 \
  --judge-gpu 0 \
  --attack-gpu-memory-utilization 0.45 \
  --target-gpu-memory-utilization 0.45
```

**Memory:** 2 models × 45% = 90% GPU usage

---

### Example 2: Each model on separate GPU
```bash
python main.py \
  --evaluate-locally \
  --use-vllm \
  --evaluate-judge-locally \
  --attack-model vicuna-13b-v1.5 \
  --target-model gemma-2-2b \
  --judge-model qwen2-57b-a14b-instruct-gptq-int4 \
  --attack-gpu 0 \
  --target-gpu 1 \
  --judge-gpu 2 \
  --attack-gpu-memory-utilization 0.85 \
  --target-gpu-memory-utilization 0.85 \
  --judge-gpu-memory-utilization 0.85
```

**Memory:** Each GPU uses 85% (no sharing)

---

### Example 3: Attack+Target on GPU 0, Judge on GPU 1
```bash
python main.py \
  --evaluate-locally \
  --use-vllm \
  --evaluate-judge-locally \
  --attack-model vicuna-13b-v1.5 \
  --target-model gemma-2-2b \
  --judge-model qwen2-57b-a14b-instruct-gptq-int4 \
  --attack-gpu 0 \
  --target-gpu 0 \
  --judge-gpu 1 \
  --attack-gpu-memory-utilization 0.45 \
  --target-gpu-memory-utilization 0.45 \
  --judge-gpu-memory-utilization 0.85
```

**Memory:** 
- GPU 0: 45% + 45% = 90%
- GPU 1: 85%

---

### Example 4: Three models on same GPU (requires lower utilization)
```bash
python main.py \
  --evaluate-locally \
  --use-vllm \
  --evaluate-judge-locally \
  --attack-model vicuna-13b-v1.5 \
  --target-model gemma-2-2b \
  --judge-model qwen2-57b-a14b-instruct-gptq-int4 \
  --attack-gpu 0 \
  --target-gpu 0 \
  --judge-gpu 0 \
  --attack-gpu-memory-utilization 0.30 \
  --target-gpu-memory-utilization 0.30 \
  --judge-gpu-memory-utilization 0.30
```

**Memory:** 3 models × 30% = 90% GPU usage

---

### Example 5: Multi-GPU for single large model
```bash
python main.py \
  --evaluate-locally \
  --use-vllm \
  --attack-model mixtral \
  --target-model gemma-2-2b \
  --attack-gpu 0,1,2,3 \
  --target-gpu 0 \
  --attack-gpu-memory-utilization 0.85 \
  --target-gpu-memory-utilization 0.45
```

**Memory:**
- GPUs 0,1,2,3: Shared for attack model (tensor/pipeline parallel)
- GPU 0: Also hosts target model at 45%

---

### Example 6: Mixed API and local execution
```bash
python main.py \
  --evaluate-locally \
  --use-vllm \
  --attack-model gpt-4-0125-preview \
  --target-model gemma-2-2b \
  --judge-model no-judge \
  --target-gpu 0 \
  --target-gpu-memory-utilization 0.85
```

**Notes:**
- Attack uses OpenAI API (no GPU)
- Target uses GPU 0 at 85%
- Judge disabled

---

## How It Works

### Internal Implementation

The system uses `CUDA_VISIBLE_DEVICES` to isolate each model to its specified GPU(s):

```python
# Before loading attack model
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Before loading target model  
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# Before loading judge model
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
```

This ensures each model only "sees" its assigned GPU(s), preventing conflicts.

### Memory Management

vLLM's `gpu_memory_utilization` parameter controls how much of the **visible GPU** memory to use:

```python
vllm.LLM(
    model=model_path,
    gpu_memory_utilization=0.45  # Use 45% of visible GPU
)
```

When multiple models share a GPU, their utilizations are **additive**.

---

## Troubleshooting

### Out of Memory (OOM) Errors

**Error:**
```
ValueError: Free memory on device (X.XX/XX.XX GiB) on startup is less than 
desired GPU memory utilization (0.9, XX.XX GiB)
```

**Solutions:**
1. **Reduce memory utilization** for models on the same GPU:
   ```bash
   --attack-gpu-memory-utilization 0.30 \
   --target-gpu-memory-utilization 0.30
   ```

2. **Separate models to different GPUs:**
   ```bash
   --attack-gpu 0 \
   --target-gpu 1
   ```

3. **Use smaller models** (e.g., 2B instead of 13B)

4. **Disable judge or use API judge:**
   ```bash
   --judge-model no-judge
   ```

---

### Judge Not Running Locally

If judge still uses API despite `--evaluate-judge-locally`:

1. **Check judge model name** - only `qwen2-57b-a14b-instruct-gptq-int4` currently supports local execution
2. **Verify flag is set:** `--evaluate-judge-locally`
3. **Check logs** for "Loaded local judge model" message

---

### GPU Not Found

**Error:**
```
CUDA error: invalid device ordinal
```

**Solution:** Check available GPUs with:
```bash
nvidia-smi
```

Only use GPU IDs that exist on your system.

---

## Performance Tips

### 1. GPU Utilization Recommendations

| Scenario | Recommended Utilization |
|----------|------------------------|
| Single model per GPU | 0.85-0.90 |
| Two models per GPU | 0.40-0.45 each |
| Three models per GPU | 0.28-0.30 each |

### 2. Model Size Considerations

Smaller models benefit more from sharing GPUs:
- **2B models:** Can fit 3-4 on a single 80GB GPU at 0.20 each
- **7B models:** 2-3 per 80GB GPU at 0.30 each
- **13B models:** 2 per 80GB GPU at 0.45 each
- **57B models (quantized):** 1-2 per 80GB GPU

### 3. Batch Size Impact

Lower GPU utilization = smaller KV cache = smaller max batch size

Reduce `--n-streams` if you get OOM during inference:
```bash
--n-streams 1  # Smallest batch size
```

---

## Default Behavior

If you don't specify GPU arguments:
- All models use **GPU 0**
- Memory utilization: **45% per model**
- Judge uses **API** (not local)

This default assumes 2 models (attack + target) on the same GPU.

---

## Migration from Previous Version

**Before:**
```bash
python main.py \
  --evaluate-locally \
  --use-vllm \
  --attack-model vicuna-13b-v1.5 \
  --target-model gemma-2-2b
```

**After (same behavior):**
```bash
python main.py \
  --evaluate-locally \
  --use-vllm \
  --attack-model vicuna-13b-v1.5 \
  --target-model gemma-2-2b \
  --attack-gpu 0 \
  --target-gpu 0 \
  --attack-gpu-memory-utilization 0.45 \
  --target-gpu-memory-utilization 0.45
```

All previous commands still work with the same default behavior!

---

## Summary

✅ **Flexible GPU assignment** for any model combination  
✅ **Transparent CLI interface** - no code changes needed  
✅ **Memory utilization control** to prevent OOM  
✅ **Local judge support** with vLLM  
✅ **Multi-GPU support** for large models  
✅ **Backward compatible** with existing commands

For more examples, see `LOCAL_MODELS_GUIDE.md`.
