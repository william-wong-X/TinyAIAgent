[🇨🇳 中文说明](README_CN.md)

# Tiny AI Agent

A lightweight AI Agent service built with **LangChain**, **LangGraph**, and **FastAPI**.

The project runs open-source LLMs fully locally via **Transformers**, exposing them through OpenAI-compatible HTTP APIs. A stateful LangGraph agent (with tool execution) consumes those APIs. It is configured with **Qwen3** by default and ships with RAG (Retrieval-Augmented Generation) plus file read/write tools.

## Architecture

Three processes cooperate at runtime:

*   **`llm/llm_api.py`** — OpenAI-compatible chat API on port **8000**. Loads the Qwen3 chat model at startup with `local_files_only=True`.
*   **`llm/embedding_api.py`** — OpenAI-compatible embeddings API on port **8001**. Loads the embedding model the same way.
*   **`main.py`** — LangGraph agent (stateful, `MemorySaver` checkpointer) with a CLI. It is a plain HTTP client: `app/model_client.py` calls the two APIs above via `ChatOpenAI`/`OpenAIEmbeddings`.

## Features

*   **Local Inference**: models run on your own machine; no external API calls, no data leaves your environment.
*   **Graph-based Agent**: LangGraph manages agent state and cyclic tool execution.
*   **RAG Ready**: incremental document sync into a Chroma vector store, queried through the `rag_search` tool.
*   **File Tools**: `read_file` reads any local file (pdf/txt/md/html/docx/ppt/csv/json...), `write_file` writes text files inside the project directory.
*   **Modular & Extensible**: add new tools in `app/tools/` and register them in `app/tools/registry.py`.

## Getting Started

### 1. Prerequisites

Python 3.10+ and the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Download the Models

Both APIs load with `local_files_only=True`, so the weights **must already exist on disk** before starting:

*   LLM: `llm/models/Qwen3-8B`
*   Embedding: `llm/models/Qwen3-Embedding-0.6B`

Download them from Hugging Face or ModelScope into those directories (paths are set in `config/config.yaml`).

### 3. Run the Agent

```bash
script/run_agent.sh
```

This starts the LLM API (8000) and the embedding API (8001) via nohup (PIDs in `pids/`, logs in `logs/`), then launches the interactive CLI (`python3 main.py`).

### 4. Build the RAG Index (optional)

Drop documents into `data/docs/` (supported: pdf/txt/md/html/docx/doc/ppt/pptx/csv/json), then:

```bash
script/build_rag_docs.sh
```

This runs an incremental sync: changes are detected via a sqlite manifest at `data/manifest/`, and chunks are upserted into Chroma at `data/vectorstores/`.

Stop all background services with `script/stop_all.sh`. Individual services can be run directly with `python3 -m llm.llm_api` and `python3 -m llm.embedding_api`. All commands accept `-c/--config <path>` to override the default `./config/config.yaml`.

## Tools

*   **`rag_search`** — retrieve relevant context from the knowledge base.
*   **`read_file`** — read any local file and return its text (with page/character limits).
*   **`write_file`** — write text to a file inside the project directory (create/overwrite/append).

## Extension Guide

1.  Define a new LangChain `Tool` in `app/tools/`.
2.  Import it and append it to the `create_tools` list in `app/tools/registry.py`.
3.  Restart the service.

Note: tool names, descriptions, argument schemas, and returned strings must be in **English** (the model reads them); UI/console strings are Chinese.

## License

[MIT](LICENSE)
