#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$PROJECT_ROOT"

mkdir -p logs pids

start_service() {
    CMD=$1
    LOG_FILE=$2
    PID_FILE=$3
    NAME=$4
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "$NAME is aleady running."
    else
        [ -f "$PID_FILE" ] && rm "$PID_FILE"
        echo "Strating $NAME..."
        nohup $CMD > "logs/$LOG_FILE" 2>&1 &
        echo $! > "$PID_FILE"
        echo "$NAME started with PID (PID: $!)."
    fi
}

start_service "python3 -m llm.embedding_api" "embedding.log" "pids/embedding.pid" "Embedding API"

echo "Waiting for Embedding API to start..."
sleep 5

python3 -m app.rag.manage_docs
echo "Document management completed."