from typing import List
from langchain_core.tools import BaseTool

from config.config import AppConfig

from app.rag.rag_service import create_rag_service
from .rag_tool import RAGSearchTool

def create_tools(config: AppConfig) -> List[BaseTool]:
    tools = []

    # rag
    db_cfg = config.vectordb.chroma
    rag_service = create_rag_service(config, db_cfg)
    rag_tool = RAGSearchTool(rag_service)
    tools.append(rag_tool)

    return tools