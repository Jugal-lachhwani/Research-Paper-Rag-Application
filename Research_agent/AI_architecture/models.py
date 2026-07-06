import os
import time
import logging
from dataclasses import dataclass
from threading import Lock
from typing import Any

from Research_agent.config import (
    embedding_model,
    qdrant,
    small_llm_config,
    vlm_model,
)

logger = logging.getLogger(__name__)

@dataclass
class AgentResources:
    small_llm: Any
    vlm: Any
    qdrant_client: Any
    dense_model: Any
    bm25_model: Any


_resources: AgentResources | None = None
_resources_lock = Lock()


def get_resources() -> AgentResources:
    global _resources

    if _resources is not None:
        return _resources

    with _resources_lock:
        if _resources is not None:
            return _resources

        if "GROQ_API_KEY" not in os.environ:
            raise RuntimeError("GROQ_API_KEY not found in environment or .env file.")

        logger.info("Initializing models (Groq LLMs, Local Embeddings)... this may take a few seconds.")
        start_time = time.time()

        from fastembed import SparseTextEmbedding
        from langchain_groq import ChatGroq
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        small_llm = ChatGroq(
            model_name=small_llm_config.MODEL_NAME,
            temperature=0.0,
        )
        vlm = ChatGroq(
            model_name=vlm_model.VLM_MODEl,
            temperature=0.0,
        )
        qdrant_client = QdrantClient(
            host=qdrant.HOST,
            port=qdrant.PORT,
        )
        dense_model = SentenceTransformer(
            embedding_model.dense_text_model,
            device="cpu",
        )
        bm25_model = SparseTextEmbedding(
            model_name=embedding_model.sparse_text_model,
        )

        _resources = AgentResources(
            small_llm=small_llm,
            vlm=vlm,
            qdrant_client=qdrant_client,
            dense_model=dense_model,
            bm25_model=bm25_model,
        )

        logger.info(f"Models initialized in {time.time() - start_time:.2f}s")
        return _resources
