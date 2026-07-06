from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List
import os
import logging
import time
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

from Research_agent.AI_architecture.prompts import (
    EXTRACTION_PROMPT,
    GENERATE_DIRECT_PROMPT,
    GENERATE_WITH_CONTEXT_PROMPT,
    RELEVANCE_PROMPT,
    SUMMARY_PROMPT,
)
from Research_agent.AI_architecture.state import GraphState
from Research_agent.AI_architecture.models import get_resources
from Research_agent.config import app_config, qdrant


def need_extraction(state: GraphState) -> Dict[str, str]:
    logger.info("--- [LOG] Node 'need_extraction' started ---")
    start_time = time.time()
    query = state.get("query", "")
    resources = get_resources()

    prompt_value = EXTRACTION_PROMPT.format(query=query)
    try:
        content = _invoke_content(resources.small_llm, prompt_value).lower()
    except Exception as exc:
        logger.error(f"Error in need_extraction calling small_llm: {exc}")
        extraction_needed = _fallback_extraction_decision(query)
        logger.warning(f"need_extraction failed, using fallback. Decision: {extraction_needed} (Latency: {time.time() - start_time:.2f}s)")
        return {"extraction_needed": extraction_needed}

    extraction_needed = "Yes" if "yes" in content else "No"
    logger.info(f"--- [LOG] Node 'need_extraction' finished. Decision: {extraction_needed} (Latency: {time.time() - start_time:.2f}s) ---")
    return {"extraction_needed": extraction_needed}

def retrieve_docs(state: GraphState) -> Dict[str, List[Dict[str, Any]]]:
    logger.info("--- [LOG] Node 'retrieve_docs' started ---")
    start_time = time.time()
    from qdrant_client import models

    query = state.get("query", "")
    resources = get_resources()

    dense_query_vector = resources.dense_model.encode(query)
    sparse_emb = list(resources.bm25_model.embed([query]))[0]
    sparse_query_vector = models.SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist(),
    )

    dense_res = resources.qdrant_client.query_points(
        collection_name=qdrant.TEXT_COLLECTION_NAME,
        query=dense_query_vector,
        using="dense",
        limit=qdrant.SEARCH_LIMIT,
        with_payload=True,
    ).points

    sparse_res = resources.qdrant_client.query_points(
        collection_name=qdrant.TEXT_COLLECTION_NAME,
        query=sparse_query_vector,
        using="sparse",
        limit=qdrant.SEARCH_LIMIT,
        with_payload=True,
    ).points

    retrieved_docs = _weighted_fusion_payloads(
        dense_res=dense_res,
        sparse_res=sparse_res,
        alpha=qdrant.ALPHA_VALUE,
        limit=qdrant.FINAL_DOC_LIMIT,
    )
    
    logger.info(f"--- [LOG] Node 'retrieve_docs' finished. Retrieved {len(retrieved_docs)} docs (Latency: {time.time() - start_time:.2f}s) ---")

    return {"retrieved_docs": retrieved_docs}


def summarize_context_tool(context: str) -> str:
    logger.info(f"Summarizing context of length {len(context)}")
    start_time = time.time()
    resources = get_resources()
    prompt_value = SUMMARY_PROMPT.format(context=context)
    response = resources.small_llm.invoke(prompt_value)
    logger.info(f"Summarization finished (Latency: {time.time() - start_time:.2f}s)")
    return response.content.strip()


def most_relevant(state: GraphState) -> Dict[str, List[Dict[str, Any]]]:
    logger.info("--- [LOG] Node 'most_relevant' started ---")
    start_time = time.time()
    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])
    resources = get_resources()

    relevant = []
    references = []
    for doc in docs:
        content = doc.get("text", "")
        if not content:
            continue

        prompt_value = RELEVANCE_PROMPT.format(
            query=query,
            document_content=content,
        )
        try:
            content_resp = _invoke_content(resources.small_llm, prompt_value).lower()
        except Exception as exc:
            logger.error(f"Error in most_relevant calling small_llm for a doc: {exc}")
            relevant.append(doc)
            paper_id = doc.get("paperid") or doc.get("paper_id") or doc.get("paperId")
            if paper_id and paper_id not in references:
                references.append(paper_id)
            continue

        if "yes" in content_resp:
            relevant.append(doc)
            paper_id = doc.get("paperid") or doc.get("paper_id") or doc.get("paperId")
            if paper_id and paper_id not in references:
                references.append(paper_id)

    logger.info(f"--- [LOG] Node 'most_relevant' finished. Filtered down to {len(relevant)} relevant docs, found {len(references)} references (Latency: {time.time() - start_time:.2f}s) ---")
    return {"relevant_docs": relevant, "references": references}


def web_search_arxiv(state: GraphState) -> Dict[str, Any]:
    logger.info("--- [LOG] Node 'web_search_arxiv' started ---")
    start_time = time.time()
    
    from Research_agent.AI_architecture.arxiv_utils import process_arxiv_fallback
    
    query = state.get("query", "")
    resources = get_resources()
    
    result = process_arxiv_fallback(
        query=query,
        small_llm=resources.small_llm,
        dense_model=resources.dense_model,
        bm25_model=resources.bm25_model
    )
    
    updates = {}
    if result:
        updates["relevant_docs"] = result.get("relevant_docs", [])
        
        # Merge references safely
        existing_refs = state.get("references", [])
        for ref in result.get("references", []):
            if ref not in existing_refs:
                existing_refs.append(ref)
        updates["references"] = existing_refs
        
        updates["new_docs_to_store"] = result.get("new_docs_to_store", [])
    
    logger.info(f"--- [LOG] Node 'web_search_arxiv' finished (Latency: {time.time() - start_time:.2f}s) ---")
    return updates

def store_in_qdrant(state: GraphState) -> Dict[str, Any]:
    logger.info("--- [LOG] Node 'store_in_qdrant' started ---")
    start_time = time.time()
    
    new_docs = state.get("new_docs_to_store", [])
    if not new_docs:
        logger.info("No new docs to store. Skipping.")
        return {}
        
    resources = get_resources()
    
    try:
        from qdrant_client import models
        points = []
        for doc in new_docs:
            dense_vec = doc["vector"]["dense"]
            bm25_vec = doc["vector"]["bm25"]
            
            points.append(
                models.PointStruct(
                    id=doc["id"],
                    payload=doc["payload"],
                    vector={
                        "dense": dense_vec,
                        "bm25": models.SparseVector(
                            indices=bm25_vec["indices"],
                            values=bm25_vec["values"]
                        )
                    }
                )
            )
            
        resources.qdrant_client.upload_points(
            collection_name=qdrant.TEXT_COLLECTION_NAME,
            points=points
        )
        logger.info(f"Successfully stored {len(points)} new points into Qdrant.")
    except Exception as exc:
        logger.error(f"Error storing in Qdrant: {exc}")
        
    logger.info(f"--- [LOG] Node 'store_in_qdrant' finished (Latency: {time.time() - start_time:.2f}s) ---")
    return {}


def generate_final_ans(state: GraphState) -> Dict[str, List[Dict[str, str]]]:
    logger.info("--- [LOG] Node 'generate_final_ans' started ---")
    start_time = time.time()
    query = state.get("query", "")
    messages = state.get("messages", [])
    relevant_docs = state.get("relevant_docs", [])
    extraction_needed = state.get("extraction_needed")
    resources = get_resources()

    if extraction_needed == "No":
        system_content = "You are a helpful and intelligent assistant. Please answer the user's question directly and concisely."
    else:
        context = "\n---\n".join(doc.get("text", "") for doc in relevant_docs)
        if _estimate_token_count(context) > app_config.SUMMARY_TOKEN_LIMIT:
            context = summarize_context_tool(context)
        
        system_content = f"You are an expert research assistant. Answer the user's question based strictly on the provided context below.\nIf the context does not contain enough information to answer the question, state that you do not have enough information.\n\nContext:\n{context}"

    langchain_messages = [SystemMessage(content=system_content)]
    
    for msg in messages:
        if msg.get("role") == "user":
            langchain_messages.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            langchain_messages.append(AIMessage(content=msg.get("content", "")))

    try:
        ans = _invoke_messages(resources.vlm, langchain_messages)
    except Exception as exc:
        logger.error(f"Error in generate_final_ans: {exc}")
        ans = _llm_failure_message(exc)

    logger.info(f"--- [LOG] Node 'generate_final_ans' finished (Latency: {time.time() - start_time:.2f}s) ---")
    return {"messages": [{"role": "assistant", "content": ans}]}


def _invoke_content(model: Any, prompt: str) -> str:
    response = model.invoke(prompt)
    return response.content.strip()


def _invoke_messages(model: Any, messages: List[Any]) -> str:
    response = model.invoke(messages)
    return response.content.strip()


def _fallback_extraction_decision(query: str) -> str:
    normalized = query.strip().lower()
    simple_phrases = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
    }

    if normalized in simple_phrases:
        return "No"

    research_terms = (
        "paper",
        "research",
        "study",
        "dataset",
        "citation",
        "method",
        "approach",
        "experiment",
        "result",
        "rmse",
        "accuracy",
        "model",
        "architecture",
    )

    if any(term in normalized for term in research_terms):
        return "Yes"

    return "Yes" if len(normalized.split()) > 4 else "No"


def _llm_failure_message(exc: Exception) -> str:
    return (
        "The NVIDIA model request failed while generating the answer. "
        "Please retry the same chat message in a moment. "
        f"Error: {type(exc).__name__}: {exc}"
    )


def _estimate_token_count(text: str) -> int:
    return max(len(text) // 4, len(text.split()))


def route_after_extraction(state: GraphState) -> str:
    logger.info("--- [LOG] Router 'route_after_extraction' evaluated ---")
    if state.get("extraction_needed") == "Yes":
        return "retrieve_docs"

    return "generate_final_ans"


def route_after_relevance(state: GraphState) -> str:
    logger.info("--- [LOG] Router 'route_after_relevance' evaluated ---")
    if not state.get("relevant_docs"):
        return "web_search_arxiv"

    return "generate_final_ans"


def _weighted_fusion_payloads(
    dense_res: List[Any],
    sparse_res: List[Any],
    alpha: float,
    limit: int,
) -> List[Dict[str, Any]]:
    dense_norm = _normalize_points(dense_res)
    sparse_norm = _normalize_points(sparse_res)
    all_doc_ids = set(dense_norm).union(sparse_norm)

    final_scores = []
    for doc_id in all_doc_ids:
        dense_data = dense_norm.get(doc_id, {"score": 0.0, "payload": None})
        sparse_data = sparse_norm.get(doc_id, {"score": 0.0, "payload": None})

        score = (
            alpha * dense_data["score"]
            + (1 - alpha) * sparse_data["score"]
        )
        payload = dense_data["payload"] or sparse_data["payload"]
        final_scores.append((score, payload))

    final_scores.sort(key=lambda item: item[0], reverse=True)
    return [
        payload
        for _, payload in final_scores[:limit]
        if payload is not None
    ]


def _normalize_points(points: List[Any]) -> Dict[str, Dict[str, Any]]:
    scores = [point.score for point in points]
    if not scores:
        return {}

    min_score = min(scores)
    max_score = max(scores)
    normalized = {}

    for point in points:
        payload = point.payload or {}
        doc_id = str(payload.get("section_id", point.id))
        score = (
            (point.score - min_score) / (max_score - min_score)
            if max_score > min_score
            else 1.0
        )

        if doc_id not in normalized or score > normalized[doc_id]["score"]:
            normalized[doc_id] = {
                "score": score,
                "payload": payload,
            }

    return normalized
