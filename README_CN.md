[🇺🇸 English](README.md)

# Tiny AI Agent

这是一个基于 **LangChain**、**LangGraph** 和 **FastAPI** 构建的轻量级 AI Agent 服务。

本项目旨在提供一个灵活的框架，用于调用开源 LLM 模型（通过 **Transformers** 加载），并能够通过 Agent 逻辑执行特定任务。目前默认配置了 RAG（检索增强生成）工具，支持根据上下文回答问题。

## 技术栈

*   **FastAPI**: 提供高性能的 HTTP API 服务。
*   **LangChain**: 用于构建 LLM 应用的编排框架。
*   **LangGraph**: 用于构建有状态、多角色的 Agent 工作流。
*   **Transformers**: 直接加载和运行开源模型。
*   **LLM Model**: 默认配置为 **Qwen3** (通义千问) 系列模型。

## 功能特性

*   **开源模型支持**: 本地化部署，默认集成 Qwen3 模型，保护数据隐私。
*   **Agent 架构**: 使用 LangGraph 构建图结构的 Agent，支持循环和条件判断。
*   **RAG 能力**: 目前内置 RAG 工具，支持知识库问答。
*   **易于扩展**: 模块化设计，可轻松添加 Search、Calculator 等自定义工具。

## 快速开始

### 1. 环境准备

确保你的 Python 版本 >= 3.10，并安装依赖：

```bash
pip install -r requirements.txt
```

### 2. 模型配置

本项目默认使用 **Qwen3**。请确保你已下载模型权重或有自动下载权限。
你可以在配置文件（如 `config.py` 或 `.env`）中修改模型路径。

### 3. 启动服务

本项目提供了启动脚本，位于 `script` 目录下。

## 扩展指南

目前 Agent 仅挂载了 RAG 工具。如果你需要添加更多工具（例如网络搜索或代码执行），请按照以下步骤操作：

1.  在 `app/tools/` 目录下定义新的 LangChain Tool。
2.  在 `app/tools/registry.py` 中，将新工具添加到 `create_tools` 列表中。
3.  修改 `config`。
4.  重启服务即可生效。

## License

[MIT](LICENSE)