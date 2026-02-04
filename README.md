[🇨🇳 中文说明](README_CN.md)

# Mini AI Agent

A lightweight AI Agent service built with **LangChain**, **LangGraph**, and **FastAPI**.

This project is designed to run open-source LLMs locally via **Transformers**. It features a stateful agent architecture capable of tool execution. By default, it is configured with the **Qwen3** model and includes a **RAG (Retrieval-Augmented Generation)** tool for context-aware QA.

## Tech Stack

*   **FastAPI**: High-performance backend API.
*   **LangChain**: Framework for LLM application orchestration.
*   **LangGraph**: For building stateful, multi-actor agent workflows.
*   **Transformers**: For loading and running open-source models locally.
*   **LLM Model**: Defaults to **Qwen3**.

## Features

*   **Local Inference**: Integrated with Hugging Face Transformers to run models without external APIs.
*   **Graph-based Agent**: Utilizes LangGraph to manage agent state and cyclic execution logic.
*   **RAG Ready**: Comes with a built-in Retrieval-Augmented Generation tool for knowledge base queries.
*   **Modular & Extensible**: Designed to be easily extended with additional tools (e.g., Search, Calculator, APIs).

## Getting Started

### 1. Prerequisites

Ensure you have Python 3.10+ installed. Install the dependencies:

```bash
pip install -r requirements.txt
```

### 2. Model Configuration

The project defaults to using **Qwen3**.
*   Ensure you have access to download the model from Hugging Face or ModelScope.
*   You can configure the model path or cache directory in the environment variables or config file.

### 3. Running the Service

Startup scripts are provided in the `script` directory.

## Extension Guide

Currently, the agent is equipped only with the **RAG tool**. To add more capabilities:

1.  Define a new LangChain `Tool` in the `app/tools/` directory.
2.  Import the tool and add it to the `create_tools` list in `app/tools/registry.py`.
3.  Modify `config`.
3.  Restart the service.

## License

[MIT](LICENSE)