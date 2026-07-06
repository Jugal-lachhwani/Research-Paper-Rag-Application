from pathlib import Path
from typing import List

from paper_qa_pipeline.abstract_ranker import AbstractTitleRanker
from paper_qa_pipeline.chunking import TextChunker
from paper_qa_pipeline.models import PaperSelectionResult, RankedPaper, TextChunk
from paper_qa_pipeline.pdf_parser import PdfParser
from research_agent.pipeline import ResearchPaperPipeline, SourceName


class PaperQAPipeline:

    def __init__(
        self,
        download_dir: str | Path = "downloads",
        search_pipeline: ResearchPaperPipeline | None = None,
        ranker: AbstractTitleRanker | None = None,
        parser: PdfParser | None = None,
        chunker: TextChunker | None = None,
    ):

        self.search_pipeline = search_pipeline or ResearchPaperPipeline(
            download_dir=download_dir,
        )
        self.ranker = ranker or AbstractTitleRanker()
        self.parser = parser or PdfParser()
        self.chunker = chunker or TextChunker()

    async def find_relevant_papers(
        self,
        query: str,
        candidate_count: int = 30,
        top_k: int = 5,
        source: SourceName = "all",
    ) -> PaperSelectionResult:

        papers = await self.search_pipeline.search(
            query=query,
            max_results=candidate_count,
            source=source,
        )

        candidates = self.ranker.rank(
            query=query,
            papers=papers,
            top_k=top_k,
        )

        return PaperSelectionResult(
            query=query,
            candidates=candidates,
        )

    async def download_candidate(
        self,
        candidate: RankedPaper,
    ) -> Path:

        return await self.search_pipeline.download(candidate.paper)

    async def download_candidates(
        self,
        candidates: List[RankedPaper],
    ) -> List[Path]:

        paths = []

        for candidate in candidates:
            if candidate.paper.pdf_url:
                paths.append(await self.download_candidate(candidate))

        return paths

    def parse_and_chunk(
        self,
        pdf_path: str | Path,
        paper_title: str,
    ) -> List[TextChunk]:

        path = Path(pdf_path)
        text = self.parser.parse(path)

        return self.chunker.chunk_text(
            text=text,
            paper_title=paper_title,
            source_path=path,
        )
