from pathlib import Path
from typing import Iterable, List, Literal, Optional

from research_agent.downloader.pdf import PdfDownloader
from research_agent.models import Paper
from research_agent.router.intent import SearchIntent
from research_agent.router.query_router import QueryRouter
from research_agent.search.arxiv_client import ArxivClient
from research_agent.search.semantic_scholar import SemanticScholarClient

SourceName = Literal["all", "arxiv", "semantic_scholar"]


class ResearchPaperPipeline:

    def __init__(
        self,
        download_dir: str | Path = "downloads",
        arxiv: Optional[ArxivClient] = None,
        semantic_scholar: Optional[SemanticScholarClient] = None,
        router: Optional[QueryRouter] = None,
    ):

        self.arxiv = arxiv or ArxivClient()
        self.semantic_scholar = semantic_scholar or SemanticScholarClient()
        self.router = router or QueryRouter()
        self.downloader = PdfDownloader(download_dir)

    async def search(
        self,
        query: str,
        max_results: int = 10,
        source: SourceName = "all",
    ) -> List[Paper]:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        intent = self.router.detect(query)

        if source == "arxiv":
            papers = await self._search_arxiv(query, intent, max_results)
        elif source == "semantic_scholar":
            papers = await self._search_semantic_scholar(query, intent, max_results)
        else:
            papers = []
            papers.extend(
                await self._search_arxiv(query, intent, max_results)
            )
            papers.extend(
                await self._search_semantic_scholar(query, intent, max_results)
            )

        papers = self._dedupe(papers)
        papers = self._rank(papers, intent)

        return papers[:max_results]

    async def download(
        self,
        paper: Paper,
        filename: Optional[str] = None,
    ) -> Path:

        return await self.downloader.download(paper, filename)

    async def download_first(
        self,
        query: str,
        max_results: int = 10,
        source: SourceName = "all",
    ) -> Path:

        papers = await self.search(query, max_results=max_results, source=source)

        for paper in papers:
            if paper.pdf_url:
                return await self.download(paper)

        raise ValueError(f"No downloadable PDF found for query: {query}")

    async def _search_arxiv(
        self,
        query: str,
        intent: SearchIntent,
        max_results: int,
    ) -> List[Paper]:

        if intent == SearchIntent.LATEST:
            return await self.arxiv.latest(query, max_results=max_results)

        return await self.arxiv.search(query, max_results=max_results)

    async def _search_semantic_scholar(
        self,
        query: str,
        intent: SearchIntent,
        max_results: int,
    ) -> List[Paper]:

        limit = max_results
        papers = await self.semantic_scholar.search(query, limit=limit)

        if intent == SearchIntent.MOST_CITED:
            return sorted(
                papers,
                key=lambda paper: paper.citation_count or 0,
                reverse=True,
            )

        return papers

    def _dedupe(self, papers: Iterable[Paper]) -> List[Paper]:

        seen = set()
        unique = []

        for paper in papers:
            key = self._dedupe_key(paper)
            if key in seen:
                continue

            seen.add(key)
            unique.append(paper)

        return unique

    def _dedupe_key(self, paper: Paper) -> str:

        if paper.doi:
            return f"doi:{paper.doi.lower()}"

        if paper.url:
            return f"url:{str(paper.url).rstrip('/')}"

        return f"title:{paper.title.lower().strip()}"

    def _rank(
        self,
        papers: List[Paper],
        intent: SearchIntent,
    ) -> List[Paper]:

        if intent == SearchIntent.MOST_CITED:
            return sorted(
                papers,
                key=lambda paper: paper.citation_count or 0,
                reverse=True,
            )

        if intent == SearchIntent.LATEST:
            return sorted(
                papers,
                key=lambda paper: (
                    paper.published.timestamp() if paper.published else 0,
                    paper.year or 0,
                ),
                reverse=True,
            )

        return sorted(
            papers,
            key=lambda paper: (
                paper.pdf_url is not None,
                paper.citation_count or 0,
                paper.year or 0,
            ),
            reverse=True,
        )
