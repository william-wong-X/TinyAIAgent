from typing import Type
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool

from app.rag.rag_service import RAGService

class RAGQueryInput(BaseModel):
    query: str = Field(description="The query string to search for in the vector database.")
    k: int = Field(default=3, description="The number of context chunks to retrieve.")

class RAGSearchTool(BaseTool):
    name: str = "rag_search"
    description: str = "Use this tool to retrieve relevant context/documents from the knowledge base."
    args_schema: Type[BaseModel] = RAGQueryInput

    _rag_service: RAGService = PrivateAttr()

    def __init__(self, rag_service: RAGService, **kwargs):
        super().__init__(**kwargs)
        self._rag_service = rag_service
    
    def _run(self, query: str, k: int = 3) -> str:
        result = self._rag_service.query_context(query, k=k)
        if result is None:
            return "No relevant information found in the knowledge base."
        return result