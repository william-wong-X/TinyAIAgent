from __future__ import annotations

import re
import json
import hashlib
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Union, List, Tuple, Sequence, Iterator, Callable, Iterable

from langchain_core.documents import Document

from config.config import PreprocessConfig, CleanConfig, SplitConfig

@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]

LoaderFn = Callable[[Path, PreprocessConfig], List[Document]]
StructureSplitterFn = Callable[[Path, List[Document], PreprocessConfig], List[Document]]
ChunkerFn = Callable[[Path, List[Document], PreprocessConfig], List[Document]]


SUPPORTED_EXTS = {
    ".pdf",
    ".txt", ".md",
    ".html", ".htm",
    ".docx", ".doc",
    ".pptx", ".ppt",
    ".csv",
    ".json",
}

def normalize_metadata(meta: Dict[str, Any], path: Path) -> Dict[str, Any]:
    m = dict(meta or {})
    m.setdefault("source", str(path))
    m.setdefault("file_name", path.name)
    m.setdefault("file_ext", path.suffix.lower())

    try:
        st = path.stat()
        m.setdefault("file_size", st.st_size)
        m.setdefault("file_mtime", int(st.st_mtime))
    except Exception:
        pass

    if "page" in m and isinstance(m["page"], int):
        m.setdefault("page_1based", m["page"] + 1)

    return m

# ====================== Scan ======================

def iter_files(inputs: Sequence[Union[str, Path]], *, recursive: bool = True) -> Iterator[Path]:
    for item in inputs:
        p = Path(item).expanduser().resolve()
        if p.is_file():
            yield p
        elif p.is_dir():
            if recursive:
                yield from (x for x in p.rglob("*") if x.is_file())
            else:
                yield from (x for x in p.glob("*") if x.is_file())

def _load_pdf(path: Path, config: PreprocessConfig) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(path), mode=config.loader.pdf_mode)
    return loader.load()

def _load_text(path: Path, config: PreprocessConfig) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(str(path), autodetect_encoding=True)
    return loader.load()

def _load_html(path: Path, config: PreprocessConfig) -> List[Document]:
    from langchain_community.document_loaders import BSHTMLLoader
    loader = BSHTMLLoader(file_path=str(path))
    return loader.load()

def _load_docx(path: Path, config: PreprocessConfig) -> List[Document]:
    if config.loader.prefer_unstructured_for_office:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
        loader = UnstructuredWordDocumentLoader(str(path), mode="single", strategy="fast")
        return loader.load()
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(str(path))
    return loader.load()

def _load_doc(path: Path, config: PreprocessConfig) -> List[Document]:
    from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    loader = UnstructuredWordDocumentLoader(str(path), mode="single", strategy="fast")
    return loader.load()

def _load_ppt(path: Path, config: PreprocessConfig) -> List[Document]:
    from langchain_community.document_loaders import UnstructuredPowerPointLoader
    loader = UnstructuredPowerPointLoader(str(path), mode="single", strategy="fast")
    return loader.load()

def _load_csv(path: Path, config: PreprocessConfig) -> List[Document]:
    from langchain_community.document_loaders import csv_loader
    loader = csv_loader.CSVLoader(str(path), autodetect_encoding=True)
    return loader.load()

def _load_json(path: Path, config: PreprocessConfig) -> List[Document]:
    raw = path.read_text(encoding="utf-8")
    try:
        obj = json.loads(raw)
    except Exception:
        return [Document(page_content=raw, metadata={"source": str(path)})]
    return [Document(page_content=raw, metadata={"source": str(path), "_json_obj": obj})]

# DEFAULT_LOADER_REGISTRY: Dict[str, LoaderFn] = {
#     ".pdf": _load_pdf,
#     ".txt": _load_text,
#     ".md": _load_text,
#     ".html": _load_html,
#     ".htm": _load_html,
#     ".docx": _load_docx,
#     ".doc": _load_doc,
#     ".ppt": _load_ppt,
#     ".pptx": _load_ppt,
#     ".csv": _load_csv,
#     ".json": _load_json
# }

DEFAULT_LOADER_REGISTRY: Dict[str, LoaderFn] = {
    ".pdf": _load_pdf,
    ".txt": _load_text,
    ".md": _load_text,
    ".html": _load_text,  # 注意：这里用 TextLoader 保留原始 HTML，便于 HTMLHeaderTextSplitter 结构切分
    ".htm": _load_text,
    ".docx": _load_docx,
    ".doc": _load_doc,
    ".ppt": _load_ppt,
    ".pptx": _load_ppt,
    ".csv": _load_csv,
    ".json": _load_json
}

# ====================== Clean ======================

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

def clean_text(text: str, config: CleanConfig) -> str:
    if config.remove_null_bytes:
        text = text.replace("\x00", "")
    
    if config.normalize_unicode_nfkc:
        text = unicodedata.normalize("NFKC", text)

    if config.normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    text =  _CONTROL_CHARS_RE.sub("", text)

    if config.dehyphenate_linebreaks:
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    if config.strip_lines:
        lines = [ln.strip() for ln in text.split("\n")]
        text = "\n".join(lines)

    if config.collapse_inline_spaces:
        text = re.sub(r"[ \t\f\v]+", " ", text)

    if config.collapse_blank_lines:
        n = max(1, config.max_consecutive_blank_lines)
        text = re.sub(r"\n{" + str(n + 1) + r",}", "\n" * n, text)
    
    return text.strip()

def clean_documents(docs: Iterable[Document], config: CleanConfig) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        t = clean_text(d.page_content or "", config)
        if not t:
            continue
        out.append(Document(page_content=t, metadata=dict(d.metadata or {})))
    return out

# ====================== Chinese ======================

DEFAULT_SEPARATORS_ZH_AWARE = [
    "\n\n",
    "\n",
    # 中文/全角句读
    "。", "！", "？", "；", "：", "…",
    "，", "、",
    # 英文句读
    ".", "!", "?", ";", ":",
    ",",
    # 空格（最后的语义弱分隔）
    " ",
    "",
]

# ====================== Split ======================

def build_recursive_splitter(config: SplitConfig):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    chunk_size = int(config.chunk_size)
    chunk_overlap = int(config.chunk_overlap)
    if chunk_overlap > chunk_size:
        chunk_overlap = max(0, chunk_size // 5)
    separators = config.separators if config.separators is not None else DEFAULT_SEPARATORS_ZH_AWARE

    if config.length_unit == "token":
        try:
            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name=config.tiktoken_encoding_name, 
                model_name=config.tiktoken_model_name, 
                chunk_size=chunk_size, 
                chunk_overlap=chunk_overlap, 
                separators=separators, 
                keep_separator=config.keep_separator, 
                add_start_index=config.add_start_index
            )
            return splitter
        except Exception:
            pass
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        keep_separator=config.keep_separator,
        add_start_index=config.add_start_index
    )
    return splitter

def default_chunker(path: Path, docs: List[Document], config: PreprocessConfig) -> List[Document]:
    splitter = build_recursive_splitter(config.split)
    return splitter.split_documents(docs)

def split_markdown_structure(path: Path, docs: List[Document], config: PreprocessConfig) -> List[Document]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    md_cfg = config.structure
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=md_cfg.markdown_headers_to_split_on, 
        strip_headers=md_cfg.markdown_strip_headers
    )
    out: List[Document] = []
    for d in docs:
        parts = splitter.split_text(d.page_content)
        for p in parts:
            meta = dict(d.metadata or {})
            meta.update(p.metadata or {})
            out.append(Document(page_content=p.page_content, metadata=meta))
    return out

def split_html_structure(path: Path, docs: List[Document], config: PreprocessConfig) -> List[Document]:
    from langchain_text_splitters import HTMLHeaderTextSplitter
    html_cfg = config.structure
    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=html_cfg.html_headers_to_split_on, 
        return_each_element=html_cfg.html_return_each_element
    )
    out: List[Document] = []
    for d in docs:
        try:
            parts = splitter.split_text(d.page_content)
        except Exception:
            parts = [d]
        for p in parts:
            meta = dict(d.metadata or {})
            meta.update(p.metadata or {})
            out.append(Document(page_content=p.page_content, metadata=meta))
    return out

def split_json_structure(path: Path, docs: List[Document], config: PreprocessConfig) -> List[Document]:
    from langchain_text_splitters.json import RecursiveJsonSplitter
    json_cfg = config.structure
    splitter = RecursiveJsonSplitter(
        max_chunk_size=config.split.chunk_size
    )
    out: List[Document] = []
    for d in docs:
        obj = d.metadata.get("_json_obj")
        if obj is None:
            try:
                obj = json.loads(d.page_content)
            except Exception:
                out.append(d)
                continue
        
        try:
            parts = splitter.create_documents(
                texts=[obj], 
                convert_lists=json_cfg.json_convert_lists, 
                ensure_ascii=False, 
                metadatas=dict(d.metadata or {})
            )
            out.extend(parts)
        except Exception:
            out.append(d)
    return out

# ====================== Preprocess ======================

class Preprocessor:
    def __init__(
            self, 
            loader_registry: Optional[Dict[str, LoaderFn]] = None, 
            structure_splitter_registry: Optional[Dict[str, StructureSplitterFn]] = None, 
            chunker_registry: Optional[Dict[str, ChunkerFn]] = None
    ):
        self.loader_registry = dict(DEFAULT_LOADER_REGISTRY if loader_registry is None else loader_registry)
        self.structure_splitter_registry: Dict[str, StructureSplitterFn] = {} if structure_splitter_registry is None else dict(structure_splitter_registry)
        self.structure_splitter_registry.setdefault(".md", split_markdown_structure)
        self.structure_splitter_registry.setdefault(".html", split_html_structure)
        self.structure_splitter_registry.setdefault(".htm", split_html_structure)
        self.structure_splitter_registry.setdefault(".json", split_json_structure)
        self.chunker_registry: Dict[str, ChunkerFn] = {} if chunker_registry is None else dict(chunker_registry)

    def register_loader(self, ext: str, fn: LoaderFn) -> None:
        self.loader_registry[ext.lower()] = fn
    
    def register_structure_splitter(self, ext: str, fn: StructureSplitterFn) -> None:
        self.structure_splitter_registry[ext.lower()] = fn

    def register_chunker(self, ext: str, fn: ChunkerFn) -> None:
        self.chunker_registry[ext.lower()] = fn
    
    def load(self, path: Path, config: PreprocessConfig) -> List[Document]:
        ext = path.suffix.lower()
        loader = self.loader_registry.get(ext)
        if loader is None:
            return []
        docs = loader(path, config)
        out: List[Document] = []
        for d in docs:
            meta = normalize_metadata(dict(d.metadata or {}), path)
            out.append(Document(page_content=d.page_content or "", metadata=meta))
        return out
    
    def structure_split(self, path: Path, docs: List[Document], config: PreprocessConfig) -> List[Document]:
        ext = path.suffix.lower()
        if ext == ".md" and not config.structure.enable_markdown_headers:
            return docs
        if ext in {".html", ".htm"} and not config.structure.enable_html_headers:
            return docs
        if ext == ".json" and not config.structure.enable_json_splitter:
            return docs
        
        splitter = self.structure_splitter_registry.get(ext)
        if splitter is None:
            return docs
        return splitter(path, docs, config)
    
    def chunk(self, path: Path, docs: List[Document], config: PreprocessConfig) -> List[Document]:
        ext = path.suffix.lower()
        chunker = self.chunker_registry.get(ext)
        if chunker is not None:
            return chunker(path, docs, config)
        return default_chunker(path, docs, config)
    
    def preprocess_file(self, file_path: Union[str, Path], *, config: PreprocessConfig) -> List[Chunk]:
        path = Path(file_path).expanduser().resolve()
        docs = self.load(path, config)
        if not docs:
            return []
        
        docs = self.structure_split(path, docs, config)

        docs = clean_documents(docs, config.clean)
        if not docs:
            return []
        
        chunk_docs = self.chunk(path, docs, config)

        chunks: List[Chunk] = []
        for i, d in enumerate(chunk_docs):
            meta = normalize_metadata(dict(d.metadata or {}), path)
            meta["chunk_index"] = i

            start = meta["start_index"]
            if isinstance(start, int):
                meta["end_index"] = start + len(d.page_content or "")
            
            if "page_1based" in meta:
                meta.setdefault("citation", f'{meta.get("file_name")}#p{meta.get("page_1based")}')
            else:
                meta.setdefault("citation", f'{meta.get("file_name")}#chunk{i}')
            
            cid = self._make_chunk_id(meta, d.page_content or "")
            chunks.append(Chunk(id=cid, text=d.page_content or "", metadata=meta))

        return chunks
    
    def preprocess(
        self,
        inputs: Sequence[Union[str, Path]],
        *,
        config: PreprocessConfig = PreprocessConfig(),
    ) -> Iterator[Chunk]:
        for fp in iter_files(inputs, recursive=config.loader.recursive):
            if config.loader.only_supported and fp.suffix.lower() not in SUPPORTED_EXTS:
                continue
            try:
                for c in self.preprocess_file(fp, config=config):
                    yield c
            except Exception as e:
                print(f"[WARN] preprocess failed: {fp} -> {type(e).__name__}: {e}")
    
    @staticmethod
    def _make_chunk_id(meta: Dict[str, Any], text: str) -> str:
        source = str(meta.get("source", ""))
        page = meta.get("page_1based", meta.get("page", ""))
        start = meta.get("start_index", "")
        th = hashlib.sha1(text.encode("utf-8")).hexdigest()
        payload = f"{source}||p={page}||s={start}||th={th}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
    
_default_pp = Preprocessor()

def preprocess(
    inputs: Sequence[Union[str, Path]],
    *,
    config: PreprocessConfig = PreprocessConfig(),
) -> Iterator[Chunk]:
    return _default_pp.preprocess(inputs, config=config)