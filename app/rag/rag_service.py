from __future__ import annotations

from typing import List

from app.rag.chromaDB import ChromaVectorStore, create_vector_store
from app.model_client import create_embedding
from app.rag.basedb import SearchResult
from config.config import AppConfig, ChromaVectorDBConfig

class RAGService:
    def __init__(self, vectorstore: ChromaVectorStore):
        self.vectorstore = vectorstore
    
    def query_context(self, query: str, k: int = 3) -> str:
        if not query:
            return ""
        
        results: List[SearchResult] = self.vectorstore.query(query_text=query, k=k)
        if not results:
            return None
        
        context_parts = []
        for i, res in enumerate(results):
            source = (res.metadata or {}).get("source", "Unknown Source")
            part = f"[Document {i+1}] (source: {source})\n{res.text}"
            context_parts.append(part)
        return "\n\n".join(context_parts)
    
def create_rag_service(app_config: AppConfig, db_config: ChromaVectorDBConfig) -> RAGService:
    embedding_fn = create_embedding(app_config).embed_documents
    vector_store = create_vector_store(db_config, embedding_fn=embedding_fn)
    return RAGService(vector_store)