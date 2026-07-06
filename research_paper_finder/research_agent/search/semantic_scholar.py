import asyncio
import time
from typing import List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
)

from research_agent.config import (
    SEMANTIC_SCHOLAR_API,
    SEMANTIC_SCHOLAR_API_KEY,
)

from research_agent.models import (
    Author,
    Paper,
)

from research_agent.utils.http import client


class AsyncRateLimiter:

    def __init__(self, min_interval_seconds: float):

        self.min_interval_seconds = min_interval_seconds
        self._lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def wait(self) -> None:

        async with self._lock:
            now = time.monotonic()
            wait_for = self.min_interval_seconds - (
                now - self._last_request_at
            )

            if wait_for > 0:
                await asyncio.sleep(wait_for)

            self._last_request_at = time.monotonic()


SEMANTIC_SCHOLAR_RATE_LIMITER = AsyncRateLimiter(
    min_interval_seconds=1.0,
)


class SemanticScholarClient:
    DEFAULT_FIELDS = (
        "title",
        "abstract",
        "authors",
        "year",
        "citationCount",
        "paperId",
        "externalIds",
        "openAccessPdf",
        "url",
    )

    def __init__(self):

        self.base_url = SEMANTIC_SCHOLAR_API

        self.headers = {}

        if SEMANTIC_SCHOLAR_API_KEY:
            self.headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2),
        reraise=True,
    )
    async def search(
        self,
        query: str,
        limit: int = 10,
        year: Optional[int] = None,
        fields_of_study: Optional[str] = None,
        fields: Optional[List[str]] = None,
    ) -> List[Paper]:

        url = f"{self.base_url}/paper/search"

        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(fields or self.DEFAULT_FIELDS),
        }

        if year:
            params["year"] = year

        if fields_of_study:
            params["fieldsOfStudy"] = fields_of_study

        await SEMANTIC_SCHOLAR_RATE_LIMITER.wait()

        response = await client.get(
            url,
            params=params,
            headers=self.headers,
        )

        response.raise_for_status()

        payload = response.json()

        papers = []

        for item in payload.get("data", []):

            pdf = item.get("openAccessPdf")
            pdf_url = pdf.get("url") if isinstance(pdf, dict) else None
            pdf_url = pdf_url or None

            papers.append(
                Paper(
                    title=item.get("title") or "Untitled paper",

                    abstract=item.get("abstract"),

                    authors=[
                        Author(
                            name=a.get("name") or "Unknown author",
                            author_id=a.get("authorId"),
                        )
                        for a in item.get("authors", [])
                    ],

                    year=item.get("year"),

                    venue=item.get("venue"),

                    citation_count=item.get("citationCount"),

                    influential_citation_count=item.get(
                        "influentialCitationCount"
                    ),

                    pdf_url=pdf_url,

                    source="Semantic Scholar",

                    paper_id=item.get("paperId"),

                    doi=(
                        item.get("externalIds", {})
                        .get("DOI")
                    ),

                    url=item.get("url") or None,

                    fields_of_study=item.get(
                        "fieldsOfStudy",
                        [],
                    ),
                )
            )

        return papers
