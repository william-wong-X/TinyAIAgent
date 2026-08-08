from typing import Type, Dict, Callable, List

from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.documents import Document

from app.rag import doc_process
from config.config import PreprocessConfig


class FileReadInput(BaseModel):
    path: str = Field(description="The absolute path of the local file to read.")
    max_pages: int = Field(
        default=20, ge=1,
        description="Maximum number of pages to read for paginated documents such as PDF; extra pages are omitted."
    )
    max_chars: int = Field(
        default=12000, ge=100,
        description="Maximum number of characters to return; the content is truncated beyond this limit."
    )


def _build_loader_registry() -> Dict[str, Callable[[Path, PreprocessConfig], List[Document]]]:
    return {
        ".pdf": doc_process._load_pdf,
        ".txt": doc_process._load_text,
        ".md": doc_process._load_text,
        ".html": doc_process._load_html,
        ".htm": doc_process._load_html,
        ".docx": doc_process._load_docx,
        ".doc": doc_process._load_doc,
        ".ppt": doc_process._load_ppt,
        ".pptx": doc_process._load_ppt,
        ".csv": doc_process._load_csv,
        ".json": doc_process._load_json,
    }


class FileReadTool(BaseTool):
    name: str = "read_file"
    description: str = (
        "Read the content of a local file and return its text. Supported formats: pdf, txt, md, html, docx, doc, ppt, pptx, csv, json, etc. "
        "For paginated documents such as PDF, the maximum number of pages can be limited; all formats support a maximum character limit."
    )
    args_schema: Type[BaseModel] = FileReadInput

    def _run(self, path: str, max_pages: int = 20, max_chars: int = 12000) -> str:
        try:
            p = Path(path).expanduser().resolve()
            if not p.is_file():
                return f"Error: file does not exist or is not a regular file: {path}"

            loaders = _build_loader_registry()
            loader = loaders.get(p.suffix.lower(), doc_process._load_text)
            config = PreprocessConfig()

            docs = loader(p, config)
        except Exception as e:
            return f"Error: failed to read file ({type(e).__name__}): {e}"

        parts: List[str] = []
        for i, d in enumerate(docs):
            text = (d.page_content or "").strip()
            if not text:
                continue

            if len(docs) > 1:
                page_no = d.metadata.get("page_1based")
                if page_no is None and isinstance(d.metadata.get("page"), int):
                    page_no = d.metadata["page"] + 1
                header = f"[Page {page_no}]" if page_no else f"[Section {i + 1}]"
                parts.append(f"{header}\n{text}")
            else:
                parts.append(text)

            if len(parts) >= max_pages:
                parts.append("...(reached max_pages, remaining content omitted)")
                break

        content = "\n\n".join(parts)

        if len(content) > max_chars:
            content = content[:max_chars] + "\n...(content exceeds max_chars, truncated)"

        return content or "(file content is empty)"
