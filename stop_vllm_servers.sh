#!/bin/bash
# Stop vLLM servers for JailbreakingLLMs PAIR experiments (ports 8004/8005/8006)

echo "========================================================================"
echo "Stopping vLLM Servers for JailbreakingLLMs"
echo "========================================================================"

cd "$(dirname "$0")"

# If we have PID files, prefer those (handles shared-model cases correctly)
PID_FILES_FOUND=false

if [ -f vllm_logs/target.pid ]; then
    TARGET_PID=$(cat vllm_logs/target.pid)
    echo "Stopping target server (PID: $TARGET_PID)..."
    kill "$TARGET_PID" 2>/dev/null || kill -9 "$TARGET_PID" 2>/dev/null || echo "  Target process already stopped"
    rm -f vllm_logs/target.pid
    PID_FILES_FOUND=true
fi

if [ -f vllm_logs/attacker.pid ]; then
    ATTACKER_PID=$(cat vllm_logs/attacker.pid)
    # Avoid killing same PID twice if models were shared
    if [ "$ATTACKER_PID" != "$TARGET_PID" ]; then
        echo "Stopping attacker server (PID: $ATTACKER_PID)..."
        kill "$ATTACKER_PID" 2>/dev/null || kill -9 "$ATTACKER_PID" 2>/dev/null || echo "  Attacker process already stopped"
    fi
    rm -f vllm_logs/attacker.pid
    PID_FILES_FOUND=true
fi

if [ -f vllm_logs/judge.pid ]; then
    JUDGE_PID=$(cat vllm_logs/judge.pid)
    if [ "$JUDGE_PID" != "$TARGET_PID" ] && [ "$JUDGE_PID" != "$ATTACKER_PID" ]; then
        echo "Stopping judge server (PID: $JUDGE_PID)..."
        kill "$JUDGE_PID" 2>/dev/null || kill -9 "$JUDGE_PID" 2>/dev/null || echo "  Judge process already stopped"
    fi
    rm -f vllm_logs/judge.pid
    PID_FILES_FOUND=true
fi

if [ "$PID_FILES_FOUND" = true ]; then
    echo ""
    echo "Stopped all servers using PID files."
    echo "========================================================================"
    exit 0
fi

echo "No PID files found. Servers may not be running, attempting to stop by port (8004, 8005, 8006)..."

for port in 8004 8005 8006; do
    pid=$(lsof -t -i:"$port" 2>/dev/null || true)
    if [ -n "$pid" ]; then
        echo "Killing process on port $port (PID: $pid)..."
        kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || echo "  Process on port $port already stopped"
    else
        echo "No process found on port $port."
    fi
done

echo ""
echo "All servers on ports 8004, 8005, 8006 should now be stopped."
echo "========================================================================"


