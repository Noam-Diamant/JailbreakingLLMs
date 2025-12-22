#!/bin/bash
# Start vLLM servers for JailbreakingLLMs PAIR experiments
# This script starts up to three vLLM servers on ports 8004, 8005, 8006.
# If some of the attack/target/judge models are the same, they will share a
# single vLLM server (first-come-first-served) and therefore a single port.
#
# Final mapping rule (must match Python mapping in main.py):
#   - Consider models in the order: TARGET, ATTACKER, JUDGE.
#   - The first unique model gets port 8004.
#   - The second unique model gets port 8005.
#   - The third unique model gets port 8006.
#   - Any repeated model reuses the already assigned port.
#
# Example:
#   TARGET_MODEL=Llama-3.1-8B, ATTACKER_MODEL=Qwen2-57B, JUDGE_MODEL=Qwen2-57B
#   -> Llama-3.1-8B on 8004, Qwen2-57B on 8005, JUDGE also uses 8005.

set -e  # Exit on error

# ============================================================================
# GPU ASSIGNMENTS - Edit or override via environment variables if needed
# ============================================================================
# Format: Single GPU "0" or multiple GPUs "0,1,2,3" for tensor/pipeline parallelism
TARGET_GPU=${TARGET_GPU:-3}       # GPU for target model  (matches --target-gpu default in main.py)
ATTACKER_GPU=${ATTACKER_GPU:-0}   # GPU for attacker model (matches --attack-gpu default in main.py)
JUDGE_GPU=${JUDGE_GPU:-0}         # GPU for judge model   (matches --judge-gpu default in main.py)

# ============================================================================
# MODEL CONFIGURATIONS - Edit or override via environment variables if needed
# ============================================================================
# Defaults chosen to match typical CLI usage in this repo.
TARGET_MODEL=${TARGET_MODEL:-"meta-llama/Llama-3.1-8B"}
ATTACKER_MODEL=${ATTACKER_MODEL:-"Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4"}
JUDGE_MODEL=${JUDGE_MODEL:-"Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4"}

# ============================================================================
# PORT CONFIGURATIONS - Fixed to 8004/8005/8006 as requested
# ============================================================================
BASE_PORTS=(8004 8005 8006)

# ============================================================================
# VLLM PARAMETERS
# ============================================================================
# GPU memory utilization (0.0 to 1.0)
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}

# Max model length (reduce if OOM)
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}

# Extra vLLM args per-model (can be overridden via env)
TARGET_EXTRA_ARGS=${TARGET_EXTRA_ARGS:-""}
ATTACKER_EXTRA_ARGS=${ATTACKER_EXTRA_ARGS:-"--quantization gptq"}
JUDGE_EXTRA_ARGS=${JUDGE_EXTRA_ARGS:-""}

# ============================================================================
# LOGGING
# ============================================================================
mkdir -p vllm_logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Prepare a simple chat template file for Llama 3.1 models if needed.
LLAMA_TEMPLATE_FILE="vllm_logs/llama3_1_chat_template.jinja"
if [ ! -f "$LLAMA_TEMPLATE_FILE" ]; then
cat > "$LLAMA_TEMPLATE_FILE" << 'EOF'
{{ bos_token }}{% for message in messages %}{% if message["role"] == "system" %}System: {{ message["content"] }}
{% elif message["role"] == "user" %}User: {{ message["content"] }}
{% elif message["role"] == "assistant" %}Assistant: {{ message["content"] }}
{% endif %}{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}
EOF
fi

# ============================================================================
# HELPERS
# ============================================================================
print_header() {
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
}

check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "ERROR: Port $1 is already in use!"
        echo "Kill the process using: kill \$(lsof -t -i:$1)"
        return 1
    fi
    return 0
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: nvidia-smi not found. Cannot verify GPU availability."
        return 0
    fi

    local gpus=$1
    IFS=',' read -ra GPU_ARRAY <<< "$gpus"
    for gpu in "${GPU_ARRAY[@]}"; do
        if ! nvidia-smi -i "$gpu" &> /dev/null; then
            echo "ERROR: GPU $gpu not found!"
            return 1
        fi
    done
    return 0
}

# ============================================================================
# DETERMINE UNIQUE MODELS AND PORT ASSIGNMENT
# This must mirror the logic in JailbreakingLLMs/main.py.
# ============================================================================

MODELS=("$TARGET_MODEL" "$ATTACKER_MODEL" "$JUDGE_MODEL")
GPUS=("$TARGET_GPU" "$ATTACKER_GPU" "$JUDGE_GPU")
ROLES=("target" "attacker" "judge")

declare -A MODEL_TO_PORT
declare -A MODEL_TO_GPU

next_port_index=0

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    gpu="${GPUS[$i]}"

    if [[ -z "${MODEL_TO_PORT[$model]+x}" ]]; then
        if (( next_port_index >= ${#BASE_PORTS[@]} )); then
            echo "ERROR: More than 3 unique models requested; only 3 ports available (8004, 8005, 8006)."
            exit 1
        fi
        port=${BASE_PORTS[$next_port_index]}
        MODEL_TO_PORT["$model"]=$port
        MODEL_TO_GPU["$model"]=$gpu
        next_port_index=$((next_port_index + 1))
    else
        # Repeated model: reuse same port, keep the GPU of the first role.
        port=${MODEL_TO_PORT[$model]}
    fi
done

TARGET_PORT=${MODEL_TO_PORT[$TARGET_MODEL]}
ATTACKER_PORT=${MODEL_TO_PORT[$ATTACKER_MODEL]}
JUDGE_PORT=${MODEL_TO_PORT[$JUDGE_MODEL]}

# ============================================================================
# MAIN
# ============================================================================
print_header "Starting vLLM Servers for JailbreakingLLMs (PAIR)"

echo "Configuration (role → model @ port on GPU):"
echo "  Target:   $TARGET_MODEL @ $TARGET_PORT on GPU(s) $TARGET_GPU"
echo "  Attacker: $ATTACKER_MODEL @ $ATTACKER_PORT on GPU(s) $ATTACKER_GPU"
echo "  Judge:    $JUDGE_MODEL @ $JUDGE_PORT on GPU(s) $JUDGE_GPU"
echo ""

echo "Unique model assignments:"
for model in "${!MODEL_TO_PORT[@]}"; do
    echo "  $model -> port ${MODEL_TO_PORT[$model]}, GPU(s) ${MODEL_TO_GPU[$model]}"
done
echo ""

# Check ports
echo "Checking ports..."
for port in "${MODEL_TO_PORT[@]}"; do
    check_port "$port" || exit 1
done
echo "All required ports are available ✓"
echo ""

# Check GPUs
echo "Checking GPUs..."
for model in "${!MODEL_TO_GPU[@]}"; do
    check_gpu "${MODEL_TO_GPU[$model]}" || exit 1
done
echo "All GPUs available ✓"
echo ""

# ============================================================================
# START SERVERS (ONE PER UNIQUE MODEL)
# ============================================================================

declare -A MODEL_TO_PID

for model in "${!MODEL_TO_PORT[@]}"; do
    port=${MODEL_TO_PORT[$model]}
    gpu=${MODEL_TO_GPU[$model]}

    # Decide extra args and log prefix by matching to role defaults
    extra_args=""
    log_prefix="model"
    if [[ "$model" == "$TARGET_MODEL" ]]; then
        extra_args="$TARGET_EXTRA_ARGS"
        log_prefix="target"
    fi

    if [[ "$model" == "$ATTACKER_MODEL" ]]; then
        extra_args="$ATTACKER_EXTRA_ARGS"
        log_prefix="attacker"
    fi
    if [[ "$model" == "$JUDGE_MODEL" ]]; then
        # If judge shares model with attacker/target, we keep whatever
        # extra_args was set first; judge-specific args only apply if
        # the judge has a distinct model.
        if [[ "$model" != "$TARGET_MODEL" && "$model" != "$ATTACKER_MODEL" ]]; then
            extra_args="$JUDGE_EXTRA_ARGS"
            log_prefix="judge"
        fi
    fi

    print_header "Starting vLLM server for model: $model"
    echo "  GPU(s): $gpu"
    echo "  Port:   $port"
    echo "  Extra:  $extra_args"

    # For Llama 3.1 models, append a simple chat template string directly via
    # --chat-template to satisfy transformers/vLLM chat requirements.
    if [[ "$model" == *"Llama-3.1-8B"* || "$model" == *"Llama-3.1-8b"* || "$model" == *"Llama-3.1"* ]]; then
        CUDA_VISIBLE_DEVICES="$gpu" nohup vllm serve "$model" \
            --port "$port" \
            --api-key EMPTY \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            $extra_args \
            --chat-template "$(cat "$LLAMA_TEMPLATE_FILE")" \
            > "vllm_logs/${log_prefix}_${TIMESTAMP}.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES="$gpu" nohup vllm serve "$model" \
            --port "$port" \
            --api-key EMPTY \
            --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
            --max-model-len "$MAX_MODEL_LEN" \
            $extra_args \
            > "vllm_logs/${log_prefix}_${TIMESTAMP}.log" 2>&1 &
    fi

    pid=$!
    MODEL_TO_PID["$model"]=$pid
    echo "Started $model (PID: $pid, log: vllm_logs/${log_prefix}_${TIMESTAMP}.log)"
    echo ""
done

# Save PIDs and ports by role for easy stopping
echo "${MODEL_TO_PID[$TARGET_MODEL]}" > vllm_logs/target.pid
echo "${MODEL_TO_PID[$ATTACKER_MODEL]}" > vllm_logs/attacker.pid
echo "${MODEL_TO_PID[$JUDGE_MODEL]}" > vllm_logs/judge.pid

echo "$TARGET_PORT" > vllm_logs/target.port
echo "$ATTACKER_PORT" > vllm_logs/attacker.port
echo "$JUDGE_PORT" > vllm_logs/judge.port

print_header "Server Status"
echo "All required servers started successfully!"
echo ""
echo "Role endpoints:"
echo "  Target:   http://localhost:$TARGET_PORT"
echo "  Attacker: http://localhost:$ATTACKER_PORT"
echo "  Judge:    http://localhost:$JUDGE_PORT"
echo ""
echo "To stop all servers:"
echo "  ./stop_vllm_servers.sh"
echo ""
echo "Check logs with, e.g.:"
echo "  tail -f vllm_logs/target_${TIMESTAMP}.log"
echo ""
echo "Waiting for servers to initialize (this may take a few minutes)..."
echo "Check logs for 'Application startup complete' message."
print_header "Setup Complete"


