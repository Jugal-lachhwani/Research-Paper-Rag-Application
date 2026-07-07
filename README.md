# 📚 Multi-Modal Research Paper RAG Application

An advanced, production-ready Retrieval-Augmented Generation (RAG) system specifically engineered for interacting with, querying, and extracting insights from ArXiv research papers. 

Built with **LangGraph** for dynamic routing, **Qdrant** for hybrid retrieval, **Redis** for semantic caching, and **Comet Opik** for prompt versioning and deep observability.

---

## ✨ Key Features & Architecture

### 🧠 LangGraph Orchestration
The core of the application is a directed cyclic graph state machine. Instead of a linear pipeline, the system uses intelligent LLM-based routers to decide whether to query the database, fallback to the web, or skip retrieval entirely based on the user's prompt.

### ⚡ Semantic Caching
To minimize latency and API costs, the system uses a **Redis-backed Semantic Cache**. When a user asks a question, the query is embedded and compared against previous questions using a vector similarity search (`COSINE` distance). If a similar query (>= 90% similarity) is found, the system instantly returns the cached response, completely bypassing the LLM and Graph.

### 🔍 Hybrid Retrieval (Qdrant)
We utilize **Qdrant** to store our chunked research papers. Rather than relying solely on semantic search, we implement **Hybrid Search**:
- **Dense Retrieval**: Captures semantic meaning using `mixedbread-ai/mxbai-embed-large-v1` (1024 dimensions).
- **Sparse Retrieval**: Captures exact keyword matches using BM25.
- **Weighted Fusion**: The results are combined using a highly optimized Weighted Fusion algorithm (Dense Weight: 0.4, Sparse Weight: 0.6) which achieved an MRR of **0.8223** during our evaluations.

### 🔄 Dynamic ArXiv Fallback (Continuous Learning)
If the user asks about a topic that isn't present in the local Qdrant database, the system doesn't just say "I don't know." It triggers a dynamic fallback:
1. Searches the ArXiv API for the top 5 related papers.
2. An LLM acts as a judge to select the single most relevant paper.
3. The system downloads the PDF, chunks the text and tables, and generates dense and sparse embeddings on the fly.
4. It dynamically stores the new chunks back into **Qdrant** so the system "learns" for future queries.

### 👁️ Observability & Prompt Versioning (Comet Opik)
Production RAG systems require deep visibility. We integrated **Comet Opik** as our observability and management layer:
- **Trace Logging**: Every LangGraph invocation, LLM call, and Semantic Cache hit/miss is deeply traced, logging latency, token usage, and retrieval metrics.
- **Prompt Versioning**: Hardcoded prompts are messy. All system prompts (Extraction, Relevancy, Final Answer) are managed, versioned, and pulled dynamically from the Opik Prompt Library (`Project_Extraction_Prompt`, etc.). This allows prompt tweaking without touching the codebase.

### ⚖️ LLM Relevancy Filter
Instead of using a computationally expensive Cross-Encoder for reranking (which increased latency by 10x in our benchmarks), we implemented an **LLM Relevancy Filter**. This layer evaluates the retrieved chunks and discards irrelevant noise before passing the context to the final generation node.

---

## 🛠️ Tech Stack

- **Orchestration**: [LangGraph](https://python.langchain.com/v0.1/docs/langgraph/)
- **LLM Inference**: [Groq](https://groq.com/) (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`)
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/)
- **Vector Database**: [Qdrant](https://qdrant.tech/)
- **Semantic Cache**: [Redis Stack](https://redis.io/)
- **Observability**: [Comet Opik](https://www.comet.com/site/products/opik/)
- **Embeddings**: 
  - Dense Text: `mixedbread-ai/mxbai-embed-large-v1` (via SentenceTransformers)
  - Sparse Text: `Qdrant/bm25` (via FastEmbed)
  - Images: `google/siglip-base-patch16-512`

---

## 🚀 Getting Started

### 1. Prerequisites
- Docker (for running Qdrant and Redis locally)
- Conda / Python 3.11+

### 2. Environment Variables
Create a `.env` file in the root directory and add the following keys:
```env
GROQ_API_KEY=your_groq_api_key
OPIK_API_KEY=your_opik_api_key
OPIK_WORKSPACE=your_opik_workspace_name
OPIK_PROJECT_NAME="Research Paper RAG Agent"
```

### 3. Start Infrastructure
Run Qdrant and Redis via Docker:
```bash
# Start Qdrant (Vector DB)
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant

# Start Redis Stack (Semantic Cache)
docker run -d --name redis -p 6379:6379 redis/redis-stack:latest
```

### 4. Sync Prompts to Opik
Upload the latest prompts to your Opik workspace:
```bash
python sync_prompts.py
```

### 5. Run the Backend API
Start the FastAPI server:
```bash
uvicorn Research_agent.api:app --host 127.0.0.1 --port 8002 --reload
```

### 6. Run the Frontend UI
In a separate terminal, serve the frontend:
```bash
cd frontend
python -m http.server 3000
```
Then navigate to `http://localhost:3000` in your browser!
