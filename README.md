# Research Paper RAG Application

An intelligent, agentic Retrieval-Augmented Generation (RAG) system specialized in academic and research papers. This application leverages LangGraph to orchestrate a complex pipeline that not only answers questions based on a local Qdrant vector database but also dynamically searches and ingests new research from arXiv when local knowledge is insufficient.

## Features

- **Agentic Workflow**: Powered by `langgraph`, the application orchestrates a multi-node pipeline evaluating if extraction is needed, retrieving documents, filtering for relevance, and gracefully falling back to web search.
- **Dynamic ArXiv Fallback**: If the local vector store lacks relevant documents for a query, the agent automatically:
  - Extracts the core topic using an SLM (Small Language Model).
  - Searches arXiv for the most relevant papers.
  - Asks the SLM to select the single most pertinent paper from the results.
  - Downloads, chunks, and embeds the PDF.
  - Directly answers the user's query and permanently stores the new document vectors in Qdrant.
- **Hybrid Search**: Uses a combination of dense embeddings (`mixedbread-ai/mxbai-embed-large-v1`) and sparse embeddings (`Qdrant/bm25`) for highly accurate retrieval.
- **FastAPI Backend**: Provides a robust REST API for seamless interaction with the agent.
- **Groq Integration**: Utilizes Groq for blazingly fast LLM inference (`llama-3.1-8b-instant`).

## Project Structure

```
├── Research_agent/
│   ├── AI_architecture/
│   │   ├── arxiv_utils.py   # ArXiv search, downloading, chunking, embedding logic
│   │   ├── graph.py         # LangGraph workflow definition and state compilation
│   │   ├── models.py        # Initializes Groq LLM and SentenceTransformer models
│   │   ├── nodes.py         # Defines all LangGraph nodes (retrieval, relevance, fallback, generation)
│   │   └── state.py         # Defines the typed dict for the graph state
│   ├── api.py               # FastAPI application and endpoints
│   ├── config.py            # Configuration variables and Qdrant constants
│   └── prompts.py           # LLM system prompts
├── Experiment_pipeline/     # Notebooks for testing RAG chunking/embedding strategies
├── open_ragbench/           # Benchmarking queries and dataset
└── .env                     # Environment variables (Groq API keys, etc.)
```

## Prerequisites

- Python 3.9+
- A running instance of Qdrant (local or cloud)
- [Groq API Key](https://console.groq.com/keys)

## Installation

1. **Clone the repository**
2. **Set up a virtual environment** (e.g., using conda):
   ```bash
   conda create -n ds python=3.10
   conda activate ds
   ```
3. **Install the dependencies**:
   *(Assuming a `requirements.txt` is present or using pip directly for LangChain, FastAPI, Qdrant, etc.)*
   ```bash
   pip install fastapi uvicorn langgraph langchain-groq qdrant-client sentence-transformers arxiv pypdf
   ```
4. **Configure Environment Variables**:
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

Start the FastAPI server:

```bash
uvicorn Research_agent.api:app --host 127.0.0.1 --port 8002 --reload
```

### Endpoints

- `GET /health` : Health check.
- `POST /chat` : Main endpoint to interact with the agent.

**Example Request:**
```json
{
  "message": "challenges in estimating output impedance in inverter-based grids"
}
```

## How It Works

1. **`need_extraction`**: Evaluates if the user's query actually requires document retrieval or if it's conversational.
2. **`retrieve_docs`**: Queries the Qdrant database using hybrid search (Dense + Sparse).
3. **`most_relevant`**: Filters the retrieved documents using the LLM to ensure only contextually accurate chunks are kept.
4. **`web_search_arxiv`**: Triggered only if `most_relevant` returns 0 documents. It searches arXiv, downloads a PDF, generates vectors locally, and queues them for storage.
5. **`generate_final_ans`**: Drafts a comprehensive answer based on the context (either from Qdrant or arXiv).
6. **`store_in_qdrant`**: Permanently uploads newly embedded arXiv vectors to Qdrant.

