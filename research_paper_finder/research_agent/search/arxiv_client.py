from typing import List
from urllib.parse import quote
import xml.etree.ElementTree as ET

from tenacity import retry, stop_after_attempt, wait_exponential

from research_agent.models import Paper, Author
from research_agent.config import ARXIV_API
from research_agent.utils.http import client


class ArxivClient:
    ATOM_NS = "{http://www.w3.org/2005/Atom}"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2),
        reraise=True,
    )
    async def _request(self, url: str):

        response = await client.get(url)

        response.raise_for_status()

        return response.text

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> List[Paper]:

        encoded = quote(query)

        url = (
            f"{ARXIV_API}"
            f"?search_query=all:{encoded}"
            f"&start=0"
            f"&max_results={max_results}"
        )

        xml = await self._request(url)

        return self._parse_feed(xml)

    async def latest(
        self,
        topic: str,
        max_results: int = 10,
    ) -> List[Paper]:

        encoded = quote(topic)

        url = (
            f"{ARXIV_API}"
            f"?search_query=all:{encoded}"
            f"&sortBy=submittedDate"
            f"&sortOrder=descending"
            f"&max_results={max_results}"
        )

        xml = await self._request(url)

        return self._parse_feed(xml)

    def _parse_feed(self, xml: str) -> List[Paper]:

        root = ET.fromstring(xml)
        papers = []

        for entry in root.findall(f"{self.ATOM_NS}entry"):
            title = self._text(entry, "title")
            published = self._text(entry, "published")
            abstract = self._text(entry, "summary")
            paper_url = self._text(entry, "id")
            pdf_url = self._pdf_url(entry)

            authors = [
                Author(name=name)
                for name in self._authors(entry)
                if name
            ]

            papers.append(
                Paper(
                    title=self._clean_text(title),
                    abstract=self._clean_text(abstract) if abstract else None,
                    authors=authors,
                    published=published or None,
                    year=int(published[:4]) if published else None,
                    pdf_url=pdf_url,
                    source="arXiv",
                    url=paper_url or None,
                )
            )

        return papers

    def _text(self, node: ET.Element, tag: str) -> str:

        child = node.find(f"{self.ATOM_NS}{tag}")
        return child.text.strip() if child is not None and child.text else ""

    def _authors(self, entry: ET.Element) -> List[str]:

        names = []

        for author in entry.findall(f"{self.ATOM_NS}author"):
            name = self._text(author, "name")
            if name:
                names.append(name)

        return names

    def _pdf_url(self, entry: ET.Element) -> str | None:

        for link in entry.findall(f"{self.ATOM_NS}link"):
            if link.attrib.get("type") == "application/pdf":
                return link.attrib.get("href")

        paper_url = self._text(entry, "id")
        if "/abs/" in paper_url:
            return paper_url.replace("/abs/", "/pdf/")

        return None

    def _clean_text(self, value: str) -> str:

        return " ".join(value.split())
