from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from Research_agent.AI_architecture.graph import get_graph_app
from Research_agent.config import app_config


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    chat_id: Optional[str] = None


class ChatResponse(BaseModel):
    chat_id: str
    answer: str
    extraction_needed: Optional[str] = None
    retrieved_docs_count: int = 0
    relevant_docs_count: int = 0
    messages: List[Dict[str, Any]] = Field(default_factory=list)


class ChatStateResponse(BaseModel):
    chat_id: str
    state: Dict[str, Any]


app = FastAPI(title=app_config.APP_NAME)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    chat_id = request.chat_id or str(uuid4())
    logger.info(f"Received chat request for chat_id: {chat_id}, query: {request.message}")
    graph_app = get_graph_app()
    config = {"configurable": {"thread_id": chat_id}}

    initial_state = {
        "query": request.message,
        "messages": [{"role": "user", "content": request.message}],
    }

    try:
        logger.info(f"Invoking graph for chat_id: {chat_id}")
        start_time = time.time()
        final_state = graph_app.invoke(initial_state, config=config)
        latency = time.time() - start_time
        logger.info(f"Graph invocation completed for chat_id: {chat_id} in {latency:.2f} seconds")
    except Exception as exc:
        logger.error(f"Error during graph invocation for chat_id {chat_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    messages = final_state.get("messages", [])
    answer = _latest_assistant_message(messages)
    
    logger.info(f"Returning response for chat_id: {chat_id}")

    return ChatResponse(
        chat_id=chat_id,
        answer=answer,
        extraction_needed=final_state.get("extraction_needed"),
        retrieved_docs_count=len(final_state.get("retrieved_docs", [])),
        relevant_docs_count=len(final_state.get("relevant_docs", [])),
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


def _latest_assistant_message(messages: List[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return str(message.get("content", ""))

    return ""
