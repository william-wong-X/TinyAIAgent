from typing import Type

from pathlib import Path

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALLOWED_MODES = {"write", "append"}


class FileWriteInput(BaseModel):
    path: str = Field(description="The path of the local file to write, relative to the project directory.")
    content: str = Field(description="The text content to write into the file.")
    mode: str = Field(
        default="write",
        description="Write mode: 'write' creates or overwrites the file; 'append' appends to the end, creating it if missing."
    )
    overwrite: bool = Field(
        default=True,
        description="Only used in 'write' mode. If False and the file already exists, the write is refused."
    )
    max_chars: int = Field(
        default=100000, ge=1,
        description="Maximum allowed content length in characters; the write fails if the content is longer."
    )


class FileWriteTool(BaseTool):
    name: str = "write_file"
    description: str = (
        "Write text content to a local file inside the project directory. "
        "Supports 'write' mode (create or overwrite a file, optionally refusing to overwrite) and 'append' mode (append to the end of a file). "
        "Missing parent directories are created automatically. Files outside the project directory cannot be written."
    )
    args_schema: Type[BaseModel] = FileWriteInput

    def _run(self, path: str, content: str, mode: str = "write", overwrite: bool = True, max_chars: int = 100000) -> str:
        try:
            if mode not in ALLOWED_MODES:
                return f"Error: invalid mode '{mode}', expected one of: {', '.join(sorted(ALLOWED_MODES))}"

            if len(content) > max_chars:
                return f"Error: content length {len(content)} exceeds max_chars={max_chars}"

            target = Path(path).expanduser().resolve()

            if not target.is_relative_to(PROJECT_ROOT):
                return f"Error: path is outside the project directory ({PROJECT_ROOT}): {path}"

            if target.is_dir():
                return f"Error: path is a directory, not a file: {path}"

            if mode == "write" and target.exists() and not overwrite:
                return f"Error: file already exists and overwrite=False; set overwrite=True to replace it: {path}"

            target.parent.mkdir(parents=True, exist_ok=True)

            if mode == "append":
                with target.open("a", encoding="utf-8") as f:
                    f.write(content)
                return f"Appended {len(content)} characters to {target}"
            else:
                target.write_text(content, encoding="utf-8")
                return f"Wrote {len(content)} characters to {target}"
        except Exception as e:
            return f"Error: failed to write file ({type(e).__name__}): {e}"
