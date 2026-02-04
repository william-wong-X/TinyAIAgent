#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

stop_process() {
    PID_FILE=$1
    NAME=$2

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "Stopping $NAME (PID: $PID)..."
            kill "$PID"
            rm "$PID_FILE"
            echo "$NAME stopped."
        else
            echo "$NAME is not running."
            rm "$PID_FILE"
        fi
    else
        :
    fi
}

stop_process "pids/llm.pid" "LLM API"
stop_process "pids/embedding.pid" "Embedding API"

echo "All services stopped."