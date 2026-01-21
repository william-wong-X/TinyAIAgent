from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Union, Optional, List, Tuple

import yaml

# ====================== Model ======================
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
# ====================== Model ======================

# ====================== VectorDB ======================
@dataclass
class ChromaVectorDBConfig:
    collection_name: str = "document"
    persist_directory: str = "data/vectorstores"

@dataclass
class VectorDBConfig:
    chroma: ChromaVectorDBConfig = field(default_factory=ChromaVectorDBConfig)
# ====================== VectorDB ======================

# ====================== RAG ======================
@dataclass
class CleanConfig:
    normalize_unicode_nfkc: bool = True
    remove_null_bytes: bool = True
    normalize_newlines: bool = True
    strip_lines: bool = True
    collapse_blank_lines: bool = True
    max_consecutive_blank_lines: int = 2
    collapse_inline_spaces: bool = True
    dehyphenate_linebreaks: bool = False


@dataclass
class SplitConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 150
    length_unit: str = "char"
    tiktoken_model_name: Optional[str] = None
    tiktoken_encoding_name: str = "gpt2"
    keep_separator: Union[bool, str] = "end"
    add_start_index: bool = True
    separators: Optional[List[str]] = None


@dataclass
class StructureSplitConfig:
    enable_markdown_headers: bool = True
    markdown_headers_to_split_on: List[Tuple[str, str]] = field(
        default_factory=lambda: [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")]
    )
    markdown_strip_headers: bool = True
    enable_html_headers: bool = True
    html_headers_to_split_on: List[Tuple[str, str]] = field(
        default_factory=lambda: [("h1", "h1"), ("h2", "h2"), ("h3", "h3"), ("h4", "h4")]
    )
    html_return_each_element: bool = False
    enable_json_splitter: bool = True
    json_convert_lists: bool = False

    def __post_init__(self):
        self.markdown_headers_to_split_on = [tuple(x) for x in self.markdown_headers_to_split_on]
        self.html_headers_to_split_on = [tuple(x) for x in self.html_headers_to_split_on]


@dataclass
class LoaderConfig:
    pdf_mode: str = "page"
    prefer_unstructured_for_office: bool = False
    recursive: bool = True
    only_supported: bool = True


@dataclass
class PreprocessConfig:
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    clean: CleanConfig = field(default_factory=CleanConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    structure: StructureSplitConfig = field(default_factory=StructureSplitConfig)

@dataclass
class SyncConfig:
    kb_root: Union[str, Path] = "data/docs"
    manifest_path: Union[str, Path] = "data/manifest/manifest.sqlite"
    confirm_by_sha256_on_suspect: bool = True
    embed_batch: int = 128
    dry_run: bool = False

@dataclass
class RAGConfig:
    vectorstore_path: str = "data/vectorstores/chroma_db"
    top_k: int = 4
# ====================== RAG ======================

@dataclass
class ToolsConfig:
    enable_desktop_tools: bool = False

@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vectordb: VectorDBConfig = field(default_factory=VectorDBConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

# ====================== Method ======================
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

        if is_dataclass(current_val) and isinstance(new_val, dict):
            _apply_overrides(current_val, new_val)
        else:
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