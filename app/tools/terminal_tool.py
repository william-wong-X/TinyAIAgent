from typing import Type, Optional

from pathlib import Path

import subprocess

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CommandInput(BaseModel):
    command: str = Field(description="The shell command to execute.")
    cwd: str = Field(
        default=None,
        description="The working directory for the command. Defaults to the project directory. Can be any path."
    )
    timeout: int = Field(
        default=30, ge=1,
        description="Timeout in seconds; the command is killed if it runs longer."
    )
    max_chars: int = Field(
        default=20000, ge=100,
        description="Maximum number of output characters to return; the output is truncated beyond this limit."
    )


class TerminalTool(BaseTool):
    name: str = "run_command"
    description: str = (
        "Execute a shell command (via bash) and return its combined stdout/stderr with the exit code. "
        "Runs with full privileges and no sandbox; the working directory can be any path. "
        "A timeout is enforced and the output is truncated to a maximum length. "
        "Not suitable for interactive commands."
    )
    args_schema: Type[BaseModel] = CommandInput

    def _run(self, command: str, cwd: Optional[str] = None, timeout: int = 30, max_chars: int = 20000) -> str:
        if cwd:
            workdir = str(Path(cwd).expanduser().resolve())
        else:
            workdir = str(PROJECT_ROOT)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=workdir,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired as e:
            partial = ""
            for src in (e.stdout, e.stderr):
                if src:
                    partial += src
            partial = partial.strip()
            if len(partial) > max_chars:
                partial = partial[:max_chars] + "\n...(content exceeds max_chars, truncated)"
            return f"Error: command timed out after {timeout}s" + (f"\n{partial}" if partial else "")
        except Exception as e:
            return f"Error: failed to run command ({type(e).__name__}): {e}"

        output = ((proc.stdout or "") + (proc.stderr or "")).strip()
        if len(output) > max_chars:
            output = output[:max_chars] + "\n...(content exceeds max_chars, truncated)"

        return f"Exit code: {proc.returncode}\n{output}".strip()
