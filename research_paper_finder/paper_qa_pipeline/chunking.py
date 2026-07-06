from pathlib import Path
from typing import List, Optional

from paper_qa_pipeline.models import TextChunk


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 1200,
        overlap: int = 200,
    ):

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(
        self,
        text: str,
        paper_title: str,
        source_path: Optional[Path] = None,
    ) -> List[TextChunk]:

        clean_text = " ".join(text.split())
        chunks = []
        start = 0
        index = 1

        while start < len(clean_text):
            end = min(start + self.chunk_size, len(clean_text))
            chunk_text = clean_text[start:end]

            chunks.append(
                TextChunk(
                    text=chunk_text,
                    chunk_id=f"chunk-{index}",
                    paper_title=paper_title,
                    source_path=source_path,
                )
            )

            if end == len(clean_text):
                break

            start = end - self.overlap
            index += 1

        return chunks
