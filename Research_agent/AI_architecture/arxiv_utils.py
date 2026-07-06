import os
import uuid
import logging
import requests
import arxiv
from typing import Any, Dict, List
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class ArxivSearchQuery(BaseModel):
    search_keyword: str = Field(description="The exact topic or keyword to search for.")
    wants_latest: str = Field(description="Whether the user prefers the newest paper. Answer 'Yes' or 'No'.")

class PaperSelection(BaseModel):
    selected_index: int = Field(description="The index (1 to N) of the most relevant paper.")
    reasoning: str = Field(description="Why this paper was chosen.")

def process_arxiv_fallback(query: str, small_llm: Any, dense_model: Any, bm25_model: Any) -> Dict[str, Any]:
    # 1. Ask SLM for search keyword and if latest is needed
    search_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the core topic/keyword for an academic search from the user's query. Also determine if they are asking for the 'latest', 'newest', or 'recent' papers. Respond in JSON."),
        ("human", "{query}")
    ])
    structured_search = small_llm.with_structured_output(ArxivSearchQuery)
    chain = search_prompt | structured_search
    
    try:
        search_args = chain.invoke({"query": query})
    except Exception as e:
        logger.error(f"Error extracting search query: {e}")
        return {}
        
    topic = search_args.search_keyword
    exact_query = f'all:"{topic}"'
    
    sort_criterion = arxiv.SortCriterion.Relevance
    if search_args.wants_latest.lower() == "yes":
        sort_criterion = arxiv.SortCriterion.SubmittedDate
        
    logger.info(f"Searching arXiv for '{topic}', Wants latest: {search_args.wants_latest}")
    
    client = arxiv.Client()
    search = arxiv.Search(
        query=exact_query,
        max_results=5,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending
    )
    
    results = list(client.results(search))
    if not results:
        logger.warning(f"No papers found on arXiv for exact phrase '{topic}'. Retrying with broad search...")
        broad_query = f"all:{topic}"
        broad_search = arxiv.Search(
            query=broad_query,
            max_results=5,
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending
        )
        results = list(client.results(broad_search))
        
        if not results:
            logger.warning("Still no papers found on arXiv with broad search.")
            return {}
        
    # 2. Ask SLM to pick best paper
    titles_context = ""
    for i, paper in enumerate(results, 1):
        titles_context += f"[{i}] Title: {paper.title}\n"
        
    select_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI research assistant. Given a specific topic and a list of paper titles retrieved from a search, select the single most relevant paper. Output the integer index of your choice and a brief reasoning."),
        ("human", "Topic: {topic}\n\nPapers:\n{titles}")
    ])
    structured_select = small_llm.with_structured_output(PaperSelection)
    chain_select = select_prompt | structured_select
    
    try:
        selection = chain_select.invoke({"topic": topic, "titles": titles_context})
        # Basic bounds check
        if selection.selected_index < 1 or selection.selected_index > len(results):
            best_paper = results[0]
        else:
            best_paper = results[selection.selected_index - 1]
    except Exception as e:
        logger.error(f"Error selecting best paper: {e}")
        best_paper = results[0]
        
    logger.info(f"LLM Selected Paper: {best_paper.title}")
    
    # 3. Download PDF
    pdf_path = "fallback_paper.pdf"
    try:
        response = requests.get(best_paper.pdf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        if response.status_code == 200 and response.content[:5] == b"%PDF-":
            with open(pdf_path, "wb") as f:
                f.write(response.content)
        else:
            logger.error("Failed to download a valid PDF.")
            return {}
    except Exception as e:
        logger.error(f"Failed to download PDF: {e}")
        return {}
        
    # 4. Chunk
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        chunks = splitter.split_documents(docs)
        text_chunks = [chunk.page_content for chunk in chunks]
    except Exception as e:
        logger.error(f"Error chunking PDF: {e}")
        return {}
        
    # 5. Embed
    logger.info("Generating dense and sparse embeddings...")
    dense_embeddings = dense_model.encode(text_chunks)
    bm25_embeddings = list(bm25_model.embed(text_chunks))
    
    # 6. Get Top 3 most relevant chunks
    query_prompt = f"Represent this sentence for searching relevant passages: {topic}"
    query_emb = dense_model.encode([query_prompt])
    similarities = dense_model.similarity(query_emb, dense_embeddings)
    
    top_indices = similarities[0].argsort(descending=True)[:3].tolist()
    
    relevant_docs = []
    for rank, idx in enumerate(top_indices, 1):
        relevant_docs.append({"text": text_chunks[idx]})
        
    # 7. Prepare docs for Qdrant storage
    paper_id = best_paper.entry_id.split('/')[-1]
    new_docs_to_store = []
    for idx, text in enumerate(text_chunks):
        
        dense_vec = dense_embeddings[idx].tolist()
        
        bm25_indices = bm25_embeddings[idx].indices.tolist() if hasattr(bm25_embeddings[idx].indices, 'tolist') else bm25_embeddings[idx].indices
        bm25_values = bm25_embeddings[idx].values.tolist() if hasattr(bm25_embeddings[idx].values, 'tolist') else bm25_embeddings[idx].values
        
        new_docs_to_store.append({
            "id": str(uuid.uuid4()),
            "payload": {
                "text": text,
                "paper_title": best_paper.title,
                "paper_id": paper_id
            },
            "vector": {
                "dense": dense_vec,
                "bm25": {
                    "indices": bm25_indices,
                    "values": bm25_values
                }
            }
        })
        
    # Clean up pdf
    if os.path.exists(pdf_path):
        os.remove(pdf_path)
        
    return {
        "relevant_docs": relevant_docs,
        "references": [best_paper.pdf_url],
        "new_docs_to_store": new_docs_to_store
    }
