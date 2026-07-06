import sqlite3
from functools import lru_cache

# pyrefly: ignore [missing-import]
from langgraph.checkpoint.sqlite import SqliteSaver
# pyrefly: ignore [missing-import]
from langgraph.graph import END, START, StateGraph

from Research_agent.AI_architecture.nodes import (
    generate_final_ans,
    most_relevant,
    need_extraction,
    web_search_arxiv,
    store_in_qdrant,
    retrieve_docs,
    route_after_extraction,
    route_after_relevance,
)
from Research_agent.AI_architecture.state import GraphState
from Research_agent.config import CHECKPOINT_DB_PATH


def build_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("need_extraction", need_extraction)
    workflow.add_node("retrieve_docs", retrieve_docs)
    workflow.add_node("most_relevant", most_relevant)
    workflow.add_node("web_search_arxiv", web_search_arxiv)
    workflow.add_node("generate_final_ans", generate_final_ans)
    workflow.add_node("store_in_qdrant", store_in_qdrant)

    workflow.add_edge(START, "need_extraction")

    workflow.add_conditional_edges(
        "need_extraction",
        route_after_extraction,
        {
            "retrieve_docs": "retrieve_docs",
            "generate_final_ans": "generate_final_ans",
        },
    )

    workflow.add_edge("retrieve_docs", "most_relevant")

    workflow.add_conditional_edges(
        "most_relevant",
        route_after_relevance,
        {
            "web_search_arxiv": "web_search_arxiv",
            "generate_final_ans": "generate_final_ans",
        },
    )

    workflow.add_edge("web_search_arxiv", "generate_final_ans")
    workflow.add_edge("generate_final_ans", "store_in_qdrant")
    workflow.add_edge("store_in_qdrant", END)

    return workflow


@lru_cache(maxsize=1)
def get_graph_app():
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        CHECKPOINT_DB_PATH,
        check_same_thread=False,
    )
    memory = SqliteSaver(conn)

    return build_workflow().compile(checkpointer=memory)


app = get_graph_app()
