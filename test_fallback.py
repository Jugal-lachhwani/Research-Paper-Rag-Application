import asyncio
from Research_agent.AI_architecture.models import get_resources
from Research_agent.AI_architecture.arxiv_utils import process_arxiv_fallback

def run_test():
    resources = get_resources()
    query = "challenges in estimating output impedance in inverter-based grids"
    print(f"Testing fallback with query: {query}")
    
    result = process_arxiv_fallback(
        query=query,
        small_llm=resources.small_llm,
        dense_model=resources.dense_model,
        bm25_model=resources.bm25_model
    )
    print("Result keys:", result.keys() if result else result)

if __name__ == "__main__":
    run_test()
