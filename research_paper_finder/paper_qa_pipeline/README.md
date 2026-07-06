# Paper QA Pipeline

This folder is for the end-to-end research-paper QA workflow:

1. Understand the user query.
2. Search arXiv/Semantic Scholar for 30-100 candidate papers.
3. Rank papers by title + abstract similarity to the user question.
4. Keep the top 5 candidate papers.
5. Let an LLM choose the final paper or top 2-3 papers.
6. Download the selected PDFs.
7. Parse PDFs into text.
8. Chunk the text.
9. Embed chunks and store them in Qdrant.
10. Retrieve the most useful chunks and generate the final answer.

The current implementation includes the first working slice:

- paper discovery through the existing `research_agent.pipeline.ResearchPaperPipeline`
- no-dependency title/abstract relevance ranking
- top candidate selection
- PDF download through the existing downloader
- plain-text chunking helpers

PDF parsing, embedding, Qdrant indexing, and LLM answer generation are isolated behind small modules so they can be filled in without mixing them into the search pipeline.

## Notebook Usage

```python
import sys
sys.path.insert(0, "/home/jugal/Documents/DataScience/Research-Paper-Rag-Application/research_paper_finder")

from paper_qa_pipeline import PaperQAPipeline

pipeline = PaperQAPipeline(download_dir="research_paper_finder/downloads")

result = await pipeline.find_relevant_papers(
    "How does retrieval augmented generation reduce hallucination?",
    candidate_count=30,
    top_k=5,
    source="arxiv",
)

for candidate in result.candidates:
    print(candidate.score, candidate.paper.title, candidate.paper.pdf_url)
```

Download the best currently selected paper:

```python
path = await pipeline.download_candidate(result.candidates[0])
path
```
