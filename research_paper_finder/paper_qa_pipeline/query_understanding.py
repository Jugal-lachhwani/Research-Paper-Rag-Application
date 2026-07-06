from research_agent.router.intent import SearchIntent
from research_agent.router.query_router import QueryRouter


class QueryUnderstanding:

    def __init__(self, router: QueryRouter | None = None):

        self.router = router or QueryRouter()

    def detect_intent(self, query: str) -> SearchIntent:

        return self.router.detect(query)
