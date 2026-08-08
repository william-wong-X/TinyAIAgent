[🇺🇸 English](README.md)

# Tiny AI Agent

这是一个基于 **LangChain**、**LangGraph** 和 **FastAPI** 构建的轻量级 AI Agent 服务。

本项目通过 **Transformers** 在本地运行开源 LLM，并以 OpenAI 兼容的 HTTP API 形式对外提供服务。一个基于 LangGraph 的状态化 Agent（支持工具调用）消费这些 API。默认配置 **Qwen3** 模型，内置 RAG（检索增强生成）以及文件读写工具。

## 架构

运行时由三个进程协作：

*   **`llm/llm_api.py`** — OpenAI 兼容的对话 API，端口 **8000**。启动时以 `local_files_only=True` 加载 Qwen3 对话模型。
*   **`llm/embedding_api.py`** — OpenAI 兼容的向量化 API，端口 **8001**。以同样方式加载 embedding 模型。
*   **`main.py`** — LangGraph Agent（有状态，使用 `MemorySaver` 检查点），带交互式 CLI。它本身是纯 HTTP 客户端：`app/model_client.py` 通过 `ChatOpenAI`/`OpenAIEmbeddings` 调用上述两个 API。

## 功能特性

*   **本地推理**: 模型完全运行在本机，不调用外部 API，数据不出本地。
*   **图结构 Agent**: 使用 LangGraph 管理 Agent 状态与循环工具执行。
*   **RAG 能力**: 文档增量同步进 Chroma 向量库，通过 `rag_search` 工具进行知识库问答。
*   **文件工具**: `read_file` 读取任意本地文件（pdf/txt/md/html/docx/ppt/csv/json 等），`write_file` 在项目目录内写入文本文件。
*   **易于扩展**: 在 `app/tools/` 下定义新工具，并在 `app/tools/registry.py` 中注册即可。

## 快速开始

### 1. 环境准备

Python 3.10+，并安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 下载模型

两个 API 都以 `local_files_only=True` 加载，因此**模型权重必须先存在于磁盘上**：

*   对话模型：`llm/models/Qwen3-8B`
*   向量模型：`llm/models/Qwen3-Embedding-0.6B`

从 Hugging Face 或 ModelScope 下载到上述目录（路径在 `config/config.yaml` 中配置）。

### 3. 启动 Agent

```bash
script/run_agent.sh
```

该脚本通过 nohup 启动 LLM API（8000）与 Embedding API（8001）（PID 存于 `pids/`，日志存于 `logs/`），然后进入交互式 CLI（`python3 main.py`）。

### 4. 构建 RAG 索引（可选）

将文档放入 `data/docs/`（支持 pdf/txt/md/html/docx/doc/ppt/pptx/csv/json），然后执行：

```bash
script/build_rag_docs.sh
```

这是一个增量同步：通过 `data/manifest/` 下的 sqlite 清单检测变化，并将分块 upsert 进 `data/vectorstores/` 的 Chroma 向量库。

使用 `script/stop_all.sh` 停止所有后台服务。也可直接运行 `python3 -m llm.llm_api` 和 `python3 -m llm.embedding_api`。所有命令支持 `-c/--config <path>` 覆盖默认的 `./config/config.yaml`。

## 工具

*   **`rag_search`** — 从知识库检索相关上下文。
*   **`read_file`** — 读取任意本地文件并返回文本（支持页数/字符数限制）。
*   **`write_file`** — 在项目目录内写入文本文件（创建/覆盖/追加）。

## 扩展指南

1.  在 `app/tools/` 目录下定义新的 LangChain Tool。
2.  在 `app/tools/registry.py` 中导入，并追加到 `create_tools` 列表。
3.  重启服务即可生效。

注意：工具的名称、描述、参数 schema 和返回字符串必须使用**英文**（模型会读取它们）；UI/控制台字符串使用中文。

## License

[MIT](LICENSE)
