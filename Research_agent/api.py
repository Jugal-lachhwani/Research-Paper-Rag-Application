from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import time
import os
from Research_agent.config import app_config

import sqlite3
from fastapi.middleware.cors import CORSMiddleware
from opik.integrations.langchain import OpikTracer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from Research_agent.AI_architecture.graph import get_graph_app


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    chat_id: Optional[str] = None


class ChatResponse(BaseModel):
    chat_id: str
    answer: str
    extraction_needed: Optional[str] = None
    retrieved_docs_count: int = 0
    relevant_docs_count: int = 0
    references: List[str] = Field(default_factory=list)
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class ChatStateResponse(BaseModel):
    chat_id: str
    state: Dict[str, Any]


app = FastAPI(title=app_config.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    from Research_agent.AI_architecture.models import get_resources
    from Research_agent.AI_architecture.semantic_cache import init_redis_index
    logger.info("Starting up application: Pre-loading all AI models and validating API keys...")
    try:
        get_resources()
        init_redis_index()
        logger.info("All models loaded successfully at startup!")
    except Exception as exc:
        logger.error(f"CRITICAL ERROR during model loading at startup: {exc}")
        raise


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


import opik
from opik import track

@app.post("/chat", response_model=ChatResponse)
@track(name="chat_endpoint")
def chat(request: ChatRequest) -> ChatResponse:
    from Research_agent.AI_architecture.models import get_resources
    from Research_agent.AI_architecture.semantic_cache import check_semantic_cache, store_semantic_cache, get_cache_size
    from opik import opik_context
    
    chat_id = request.chat_id or str(uuid4())
    logger.info(f"Received chat request for chat_id: {chat_id}, query: {request.message}")
    
    resources = get_resources()
    
    # 1. Semantic Caching
    start_time = time.time()
    query_emb = resources.dense_model.encode([request.message])[0]
    
    cache_hit = check_semantic_cache(query_emb)
    cache_lookup_latency = time.time() - start_time
    
    if cache_hit:
        logger.info(f"Semantic Cache HIT for chat_id: {chat_id}")
        answer = cache_hit["answer"]
        similarity = cache_hit["similarity"]
        
        try:
            opik_context.update_current_trace(metadata={
                "cache_hit": True,
                "cache_similarity_score": similarity,
                "cache_lookup_latency": cache_lookup_latency,
                "cached_entries_count": get_cache_size()
            })
        except Exception as e:
            logger.error(f"Opik logging error: {e}")
            
        messages = [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": answer}
        ]
        
        return ChatResponse(
            chat_id=chat_id,
            answer=answer,
            extraction_needed="No",
            retrieved_docs_count=0,
            relevant_docs_count=0,
            references=[],
            messages=messages,
        )

    # 2. Semantic Cache MISS - Invoke LangGraph
    logger.info(f"Semantic Cache MISS for chat_id: {chat_id}. Invoking graph.")
    graph_app = get_graph_app()
    
    opik_tracer = OpikTracer(project_name=app_config.APP_NAME)
    
    config = {
        "configurable": {"thread_id": chat_id},
        "callbacks": [opik_tracer]
    }

    initial_state = {
        "query": request.message,
        "messages": [{"role": "user", "content": request.message}],
    }

    try:
        final_state = graph_app.invoke(initial_state, config=config)
        latency = time.time() - start_time
        logger.info(f"Graph invocation completed for chat_id: {chat_id} in {latency:.2f} seconds")
    except Exception as exc:
        logger.error(f"Error during graph invocation for chat_id {chat_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    messages = final_state.get("messages", [])
    answer = _latest_assistant_message(messages)
    
    # Store result in Semantic Cache
    store_semantic_cache(request.message, answer, query_emb)
    
    try:
        opik_context.update_current_trace(metadata={
            "cache_hit": False,
            "cache_lookup_latency": cache_lookup_latency,
            "cached_entries_count": get_cache_size()
        })
    except Exception as e:
        logger.error(f"Opik logging error: {e}")
    
    logger.info(f"Returning response for chat_id: {chat_id}")

    return ChatResponse(
        chat_id=chat_id,
        answer=answer,
        extraction_needed=final_state.get("extraction_needed"),
        retrieved_docs_count=len(final_state.get("retrieved_docs", [])),
        relevant_docs_count=len(final_state.get("relevant_docs", [])),
        references=final_state.get("references", []),
        messages=messages,
    )


@app.get("/chats/{chat_id}/state", response_model=ChatStateResponse)
def get_chat_state(chat_id: str) -> ChatStateResponse:
    graph_app = get_graph_app()
    config = {"configurable": {"thread_id": chat_id}}
    snapshot = graph_app.get_state(config)

    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Chat not found.")

    return ChatStateResponse(
        chat_id=chat_id,
        state=snapshot.values,
    )


@app.get("/chats", response_model=List[str])
def get_all_chats() -> List[str]:
    """Retrieve all unique chat IDs from the SQLite checkpoints database."""
    try:
        db_path = os.path.join(os.path.dirname(__file__), "checkpoints.sqlite")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        rows = cursor.fetchall()
        chat_ids = [row[0] for row in rows if row[0]]
        conn.close()
        return chat_ids
    except Exception as exc:
        logger.error(f"Error fetching chat IDs from database: {exc}")
        raise HTTPException(status_code=500, detail="Could not fetch chat history")


def _latest_assistant_message(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))

    return ""
