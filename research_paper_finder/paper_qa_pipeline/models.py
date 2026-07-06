from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from research_agent.models import Paper


@dataclass
class RankedPaper:
    paper: Paper
    score: float
    rank: int
    reason: str


@dataclass
class PaperSelectionResult:
    query: str
    candidates: List[RankedPaper]


@dataclass
class TextChunk:
    text: str
    chunk_id: str
    paper_title: str
    source_path: Optional[Path] = None
    page: Optional[int] = None
