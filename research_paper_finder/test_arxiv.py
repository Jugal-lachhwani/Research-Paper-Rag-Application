import asyncio

from research_agent.search.arxiv_client import ArxivClient


async def main():

    client = ArxivClient()

    papers = await client.latest(
        "quantum computing",
        max_results=5,
    )

    for paper in papers:

        print("=" * 80)

        print(paper.title)

        print(paper.year)

        print(paper.pdf_url)


asyncio.run(main())