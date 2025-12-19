from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

@dataclass
class LLMConfig:
    base_url: str = "http://localhost:8000/v1"
    model: str = "qwen3-8b"
    model_path: str = "llm/models/Qwen3-8B"
    temperature: float = 0.7
    enable_thinking: bool = True
    streaming: bool = True

@dataclass
class EmbeddingConfig:
    base_url: str = "http://localhost:8001/v1"
    model: str = "qwen3-wmbedding-0.6b"
    model_path: str = "llm/models/Qwen3-Embedding-0.6B"

@dataclass
class RAGConfig:
    vectorstore_path: str = "data/vectorstores/chroma_db"
    top_k: int = 4

@dataclass
class ToolsConfig:
    enable_desktop_tools: bool = False

@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

def _load_raw_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
    
def _apply_overrides(dc: Any, overrides: Dict[str, Any]) -> Any:
    if not is_dataclass(dc):
        return dc
    
    for f in fields(dc):
        name = f.name
        if name not in overrides:
            continue

        current_val = getattr(dc, name)
        new_val = overrides[name]

        # 嵌套 dataclass：递归覆盖
        if is_dataclass(current_val) and isinstance(new_val, dict):
            _apply_overrides(current_val, new_val)
        else:
            # 简单字段，直接覆盖
            setattr(dc, name, new_val)

    return dc

@lru_cache(maxsize=None)
def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        path = os.getenv("APP_CONFIG_FILE", "config.yaml")

    raw = _load_raw_config(path)
    cfg = AppConfig()
    cfg = _apply_overrides(cfg, raw)
    return cfg