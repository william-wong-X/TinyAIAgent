from __future__ import annotations

from typing import Dict, Any, Optional, List

import uuid
import chromadb
from chromadb.api.models.Collection import Collection

from app.rag.basedb import VectorStore, SearchResult, EmbeddingFn
from config.config import ChromaVectorDBConfig

class ChromaVectorStore(VectorStore):
    def __init__(self,
                 collection_name: str, 
                 embedding_fn: EmbeddingFn, 
                 persist_directory: str = "data/vectorstores"):
        self.embedding_fn = embedding_fn
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection: Collection = self.client.get_or_create_collection(collection_name)

    def upsert_texts(self, 
                     texts: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None, 
                     ids: Optional[List[str]] = None) -> List[str]:
        if not texts:
            return []
        
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("Length of metadatas and texts must equal")

        if ids is not None and len(ids) != len(texts):
            raise ValueError("Length of ids and texts must equal")

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        
        embeddings = self.embedding_fn(texts)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return ids
    
    def query(self, 
              query_text: Optional[str] = None, 
              query_embedding: Optional[List[float]] = None, 
              k: int = 5, 
              where: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        if query_text is None:
            if query_embedding is None:
                raise ValueError("loss query_text or query_embedding")
            query_embedding = self.embedding_fn([query_text])[0]
        
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[SearchResult] = []
        for i in range(len(ids)):
            out.append(
                SearchResult(
                    id=str(ids[i]),
                    text=docs[i] if docs and i < len(docs) else "",
                    metadata=metas[i] if metas and i < len(metas) else {},
                    distance=dists[i] if dists and i < len(dists) else None,
                )
            )
        return out
    
    def delete(self, 
               ids: Optional[List[str]] = None, 
               where: Optional[Dict[str, Any]] = None) -> None:
        if ids is None and where is None:
            raise ValueError("need ids or where")
        self.collection.delete(ids=ids, where=where)

    def count(self) -> int:
        return self._collection.count()
    
    def persist(self):
        return super().persist()
    
    def close(self):
        return super().close()

def create_vector_store(cfg: ChromaVectorDBConfig, embedding_fn: EmbeddingFn) -> VectorStore:
    return ChromaVectorStore(
        collection_name=cfg.collection_name,
        persist_directory=cfg.persist_directory,
        embedding_fn=embedding_fn,
    )