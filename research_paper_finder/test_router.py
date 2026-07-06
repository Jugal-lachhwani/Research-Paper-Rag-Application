from research_agent.router.query_router import QueryRouter

router = QueryRouter()

tests = [

    "Attention Is All You Need",

    "latest research in quantum computing",

    "most cited RAG papers",

    "papers by Geoffrey Hinton",

    "LLM Agents",

]

for t in tests:

    print(t)

    print(router.detect(t))

    print()