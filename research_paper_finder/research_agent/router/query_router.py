import re

from research_agent.router.intent import SearchIntent


class QueryRouter:

    LATEST_PATTERN = re.compile(
        r"\b(latest|recent|newest|today|this year)\b",
        re.IGNORECASE,
    )

    CITATION_PATTERN = re.compile(
        r"\b(cited|citation|most influential|best papers|highly cited)\b",
        re.IGNORECASE,
    )

    AUTHOR_PATTERN = re.compile(
        r"\b(author|written by|papers by)\b",
        re.IGNORECASE,
    )

    def detect(
        self,
        query: str,
    ) -> SearchIntent:

        query = query.strip()

        if not query:
            return SearchIntent.UNKNOWN

        if self.LATEST_PATTERN.search(query):
            return SearchIntent.LATEST

        if self.CITATION_PATTERN.search(query):
            return SearchIntent.MOST_CITED

        if self.AUTHOR_PATTERN.search(query):
            return SearchIntent.AUTHOR

        if (
            len(query.split()) <= 8
            and query[0].isupper()
        ):
            return SearchIntent.EXACT_TITLE

        return SearchIntent.TOPIC
