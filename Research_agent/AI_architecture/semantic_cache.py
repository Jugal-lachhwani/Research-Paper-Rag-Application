import redis
import numpy as np
import uuid
import logging
from typing import Optional, Dict, Any
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

logger = logging.getLogger(__name__)

CACHE_INDEX_NAME = "semantic_cache_idx"
CACHE_PREFIX = "cache:"
SIMILARITY_THRESHOLD = 0.90 # Score of 0.90 similarity

r = redis.Redis(host='localhost', port=6379)

def init_redis_index(vector_dim: int = 1024):
    try:
        schema = (
            TextField("query"),
            TextField("answer"),
            VectorField("embedding", "FLAT", {
                "TYPE": "FLOAT32",
                "DIM": vector_dim,
                "DISTANCE_METRIC": "COSINE"
            }),
        )
        r.ft(CACHE_INDEX_NAME).create_index(
            schema, 
            definition=IndexDefinition(prefix=[CACHE_PREFIX], index_type=IndexType.HASH)
        )
        logger.info(f"Created Redis semantic cache index: {CACHE_INDEX_NAME}")
    except Exception as e:
        if "Index already exists" in str(e):
            logger.info("Redis semantic cache index already exists.")
        else:
            logger.warning(f"Failed to create Redis index or it exists: {e}")

def check_semantic_cache(embedding: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Search Redis for a semantically similar query.
    Returns the cached answer dictionary if similarity > threshold, else None.
    """
    try:
        emb_bytes = embedding.astype(np.float32).tobytes()
        
        q = Query(f"*=>[KNN 1 @embedding $vec AS score]").return_fields("query", "answer", "score").dialect(2)
        res = r.ft(CACHE_INDEX_NAME).search(q, {"vec": emb_bytes})
        
        if res.docs:
            doc = res.docs[0]
            # In Redis COSINE metric, score is distance (1 - similarity)
            # So similarity = 1 - distance
            distance = float(doc.score)
            similarity = 1.0 - distance
            
            logger.info(f"Cache lookup top match: '{doc.query}' with similarity {similarity:.4f}")
            
            if similarity >= SIMILARITY_THRESHOLD:
                return {
                    "query": doc.query,
                    "answer": doc.answer,
                    "similarity": similarity
                }
    except Exception as e:
        logger.error(f"Error checking semantic cache: {e}")
        
    return None

def store_semantic_cache(query: str, answer: str, embedding: np.ndarray):
    """
    Store a query, its embedding, and the answer in Redis.
    """
    try:
        doc_id = f"{CACHE_PREFIX}{uuid.uuid4()}"
        emb_bytes = embedding.astype(np.float32).tobytes()
        r.hset(doc_id, mapping={
            "query": query,
            "answer": answer,
            "embedding": emb_bytes
        })
        logger.info(f"Stored query in semantic cache: {doc_id}")
    except Exception as e:
        logger.error(f"Error storing in semantic cache: {e}")

def get_cache_size() -> int:
    try:
        info = r.ft(CACHE_INDEX_NAME).info()
        return int(info.get("num_docs", 0))
    except Exception:
        return 0
