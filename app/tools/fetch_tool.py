from typing import Type

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

ALLOWED_SCHEMES = {"http", "https"}

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TinyAIAgent/1.0; local agent)"
}


class FetchUrlInput(BaseModel):
    url: str = Field(description="The URL of the web page to fetch.")
    max_chars: int = Field(
        default=12000, ge=100,
        description="Maximum number of characters to return; the content is truncated beyond this limit."
    )


class FetchUrlTool(BaseTool):
    name: str = "fetch_url"
    description: str = (
        "Fetch a web page by URL and return its title and main text content. "
        "Only http and https URLs are supported. The returned text is truncated to a maximum length."
    )
    args_schema: Type[BaseModel] = FetchUrlInput

    def _run(self, url: str, max_chars: int = 12000) -> str:
        try:
            from urllib.parse import urlparse
            import requests
            from bs4 import BeautifulSoup
        except ImportError as e:
            return f"Error: missing dependency ({e}); install 'requests' and 'bs4'"

        try:
            parsed = urlparse(url)
            if parsed.scheme not in ALLOWED_SCHEMES:
                return f"Error: only http and https URLs are supported: {url}"

            resp = requests.get(url, timeout=15, headers=DEFAULT_HEADERS, allow_redirects=True)
            if resp.status_code != 200:
                return f"Error: request failed with HTTP {resp.status_code}: {url}"
            resp.encoding = resp.apparent_encoding

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside", "iframe"]):
                tag.decompose()

            title = soup.title.get_text(strip=True) if soup.title else ""

            text = soup.get_text(separator="\n")
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            body = "\n".join(lines)

            parts = []
            if title:
                parts.append(f"Title: {title}")
            parts.append(body)
            content = "\n\n".join(parts).strip()

            if len(content) > max_chars:
                content = content[:max_chars] + "\n...(content exceeds max_chars, truncated)"

            return content or "(page content is empty)"
        except Exception as e:
            return f"Error: failed to fetch URL ({type(e).__name__}): {e}"
