import asyncio
import time

from research_agent.search.semantic_scholar import SemanticScholarClient


async def main():

    client = SemanticScholarClient()

    if "x-api-key" not in client.headers:
        raise RuntimeError(
            "SEMANTIC_SCHOLAR_API_KEY was not loaded into the x-api-key header."
        )

    start = time.monotonic()

    papers = await client.search(
        query="retrieval augmented generation",
        limit=3,
    )

    more_papers = await client.search(
        query="attention is all you need",
        limit=3,
    )

    elapsed = time.monotonic() - start

    if elapsed < 1.0:
        raise RuntimeError("Semantic Scholar rate limiter did not wait.")

    if not papers:
        raise RuntimeError("Semantic Scholar returned no papers.")

    print("Semantic Scholar API key loaded into x-api-key header.")
    print(f"First search returned {len(papers)} papers.")
    print(f"Second search returned {len(more_papers)} papers.")
    print(f"Elapsed time for two scheduled requests: {elapsed:.2f}s")

    for paper in papers:
        print("=" * 80)
        print(paper.title)
        print(paper.year)
        print(paper.citation_count)
        print(paper.pdf_url)
    
    for paper in more_papers:
        print("=" * 80)
        print(paper.title)
        print(paper.year)
        print(paper.citation_count)
        print(paper.pdf_url)


asyncio.run(main())
