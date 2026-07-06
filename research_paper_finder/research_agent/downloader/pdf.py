from pathlib import Path
import re
from typing import Optional
from urllib.parse import urlparse

from research_agent.models import Paper
from research_agent.utils.http import client


class PdfDownloader:

    def __init__(self, download_dir: str | Path = "downloads"):

        self.download_dir = Path(download_dir)

    async def download(
        self,
        paper: Paper,
        filename: Optional[str] = None,
    ) -> Path:

        if not paper.pdf_url:
            raise ValueError(f"No PDF URL available for: {paper.title}")

        self.download_dir.mkdir(parents=True, exist_ok=True)

        target = self.download_dir / (
            filename or self._filename_for(paper)
        )

        response = await client.get(str(paper.pdf_url))
        response.raise_for_status()

        content_type = response.headers.get("content-type", "").lower()
        if "pdf" not in content_type and not str(paper.pdf_url).endswith(".pdf"):
            raise ValueError(
                f"URL did not return a PDF for '{paper.title}': {content_type}"
            )

        target.write_bytes(response.content)
        return target

    def _filename_for(self, paper: Paper) -> str:

        stem = paper.paper_id or self._slug(paper.title)
        if paper.source.lower() == "arxiv" and paper.url:
            parsed = urlparse(str(paper.url))
            arxiv_id = parsed.path.rstrip("/").split("/")[-1]
            stem = arxiv_id or stem

        return f"{self._slug(stem)}.pdf"

    def _slug(self, value: str) -> str:

        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-._")
        return slug[:120] or "paper"
