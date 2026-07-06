import asyncio

from research_agent.pipeline import ResearchPaperPipeline


async def main():

    pipeline = ResearchPaperPipeline(
        download_dir="research_paper_finder/downloads",
    )

    papers = await pipeline.search(
        query="latest research in quantum computing",
        max_results=5,
        source="arxiv",
    )

    for paper in papers:

        print("=" * 80)

        print(paper.title)

        print(paper.year)

        print(paper.pdf_url)


asyncio.run(main())
