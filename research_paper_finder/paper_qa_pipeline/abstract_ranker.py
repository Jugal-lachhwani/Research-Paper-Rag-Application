import math
import re
from collections import Counter
from typing import Iterable, List

from paper_qa_pipeline.models import RankedPaper
from research_agent.models import Paper


class AbstractTitleRanker:

    TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]*")

    def rank(
        self,
        query: str,
        papers: Iterable[Paper],
        top_k: int = 5,
    ) -> List[RankedPaper]:

        query_vector = self._vectorize(query)
        ranked = []

        for paper in papers:
            text = self._paper_text(paper)
            score = self._cosine(query_vector, self._vectorize(text))

            ranked.append(
                RankedPaper(
                    paper=paper,
                    score=score,
                    rank=0,
                    reason=self._reason(paper, score),
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)

        for index, candidate in enumerate(ranked[:top_k], start=1):
            candidate.rank = index

        return ranked[:top_k]

    def _paper_text(self, paper: Paper) -> str:

        return " ".join(
            value
            for value in [paper.title, paper.abstract or ""]
            if value
        )

    def _vectorize(self, text: str) -> Counter:

        return Counter(
            token.lower()
            for token in self.TOKEN_PATTERN.findall(text)
        )

    def _cosine(self, left: Counter, right: Counter) -> float:

        if not left or not right:
            return 0.0

        common = set(left).intersection(right)
        dot = sum(left[token] * right[token] for token in common)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return dot / (left_norm * right_norm)

    def _reason(self, paper: Paper, score: float) -> str:

        if score == 0:
            return "No direct lexical overlap with the query."

        if paper.abstract:
            return "Matched against title and abstract text."

        return "Matched against title text only."
