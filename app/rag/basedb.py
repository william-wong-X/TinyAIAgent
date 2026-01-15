from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable

EmbeddingFn = Callable[[List[str], List[List[float]]]]

@dataclass
class SearchResult:
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None

class VectorStore(ABC):
    @abstractmethod
    def upsert_texts(
        self,  
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None, 
        ids: Optional[List[str]] = None
    ) -> List[str]:
        '''write/update texts'''
    
    @abstractmethod
    def query(
        self, 
        query_text: Optional[str] = None, 
        query_embedding: Optional[List[float]] = None, 
        k: int = 5, 
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        '''search texts'''

    @abstractmethod
    def delete(
        self, 
        ids: Optional[List[str]] = None, 
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        '''delete texts'''

    @abstractmethod
    def count(self) -> int:
        """count"""

    @abstractmethod
    def persist(self) -> None:
        """persist"""

    @abstractmethod
    def close(self) -> None:
        """close"""