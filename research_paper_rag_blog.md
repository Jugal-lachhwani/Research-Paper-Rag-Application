# Building and Evaluating a Multi-Model Research Paper RAG

In this blog, I will walk you through production-level concepts for Retrieval-Augmented Generation (RAG) and how to build an end-to-end RAG pipeline. We will explore how to choose the right retrieval method, chunking strategy, embedding model, and Large Language Model (LLM).

## Step 1: Getting the Data

In any RAG system, data is the most critical component. Your data is your ground truth. Before building any solution, you must thoroughly understand the source and quality of the data you are using. The LLM uses this data as its context to answer questions; if the data quality is poor, the LLM will likely fail to give expected results, even if the rest of your pipeline is perfect.

In this project, we are using research papers sourced from ArXiv. Since data quality is paramount, let's discuss the characteristics of our data:

- The data is provided in JSON format.
- Images are converted into Base64 encoding.
- Tables are separated from the main text.
- The data is divided into sections based on headings.
- The text and tables are converted into Markdown format.

### Problems with the Data

- Because the text is converted to Markdown, formulas present in the paper are not always correctly formatted. As a result, the LLM may not interpret them correctly.
- References are generally not useful for answering queries, so we can remove them.
- Sections are not always perfectly divided, which presents challenges during the chunking phase.

## Step 2: The Chunking Strategy

Because an LLM has a limited context window and cannot process an entire document at once, we use "chunking" to feed the LLM only the relevant pieces of information. This also prevents hallucinations.

Deciding on a chunking method is the first major step in building a RAG pipeline. It depends highly on your specific data. We must ensure that the chunks are meaningful so the LLM can understand the context. If chunking is done poorly, the LLM will receive fragmented or misleading context, failing to answer the user's query accurately.

There are many basic and advanced chunking methods. We will discuss them below and decide which is right for our project.

### Simple Chunking
- Splits text blindly at a certain character limit.
- For example, if the limit is set to 2, the word "Rock" is divided into "Ro" and "ck".
- It does not account for words, sentences, or context.
- All chunks are exactly equal in length.

### Recursive Character Splitting
- Splits text based on a hierarchy of separators, such as `['\n\n', '\n', '.', ' ']`.
- It tries to keep paragraphs, sentences, and words together. While chunks are not guaranteed to be of exactly equal length, we can set a threshold so no chunk exceeds a maximum size.

### Parent-Child Chunking
- Chunking is performed in a hierarchical format.
- For example, a "parent" chunk might be 1,000 characters, and it is subdivided into "child" chunks of 200 characters each.
- The vector database retrieves the highly relevant child chunks, but the system passes the entire parent chunk to the LLM to provide broader context.

### Semantic Chunking
- This method chunks text based on semantic meaning rather than character counts.
- **Steps:** The document is split into individual sentences. The system calculates the semantic similarity between consecutive sentences; if they exceed a certain similarity threshold, they are grouped into the same chunk.
- While this sounds ideal in theory, it often falls short in practice and can be computationally expensive.

### Contextual Chunking (Anthropic's Approach)
This can be implemented in two ways:

**First Method (Less Expensive):**
- Generate a summary of the entire document.
- Divide the document into standard chunks.
- Append the document summary to the beginning of every chunk.

**Second Method (More Expensive):**
- Pass the whole document along with a specific chunk to an LLM.
- Ask the LLM to generate a short explanation of the chunk's role within the document.
- Prepend this LLM-generated context to the chunk before embedding.

### Conclusion on Chunking

Every method has advantages and disadvantages. In production, the most commonly used methods are:

- **Contextual Chunking:** Expensive but highly accurate.
- **Parent-Child Chunking:** Much less expensive than Contextual Chunking and provides good accuracy. The downside is increased retrieval latency.
- **Recursive Character Splitting:** The easiest to implement and works well for most standard tasks. Both chunking and retrieval are fast, though there is a risk of losing context at the boundaries.

This ultimately comes down to a trade-off between accuracy, speed, and cost.

### What I Used and Why

In this project, I used a hybrid approach combining **Section Splitting** and **Recursive Character Splitting**.

- First, the text is split into major sections (e.g., Abstract, Methodology) using a Markdown Splitter. References are discarded.
- Next, Recursive Character Splitting is applied to divide the sections into 1,500-character chunks with a 300-character overlap. 
- Overlapping decreases the chance of splitting a core topic down the middle. Splitting is only applied if a section exceeds 2,000 characters.
- To preserve context, the paper's title and the section's heading are appended to the start of every chunk.
- **Handling Tables:** Tables (already converted to text) are split only if they exceed 2,000 characters, using table-specific chunking techniques. The title of the table is prepended to every chunk derived from it.

## Step 3: Embedding

Because we are building a Multi-Modal RAG, text (including tables) and images are embedded separately and added to the vector store.

Choosing the right embedding model is crucial. There are many high-accuracy, open-source models available on Hugging Face. You should evaluate them using the **MTEB (Massive Text Embedding Benchmark) Leaderboard** to compare performance across different tasks and select the model that best suits your needs.

For this project, I used:
- **Text Embeddings:** `"mixedbread-ai/mxbai-embed-large-v1"` (1024 dimensions, 200M parameters).
- **Image Embeddings:** `"google/siglip-base-patch16-512"`.

## Step 4: Vector Store

A Vector Store is a specialized database designed to handle high-dimensional vectors.

### Core Capabilities

| Capability | Purpose |
|------------|---------|
| Vector storage | Persist high-dimensional embeddings |
| Similarity search | Find nearest neighbors quickly |
| Metadata filtering | Combine vector search with attribute filters |
| CRUD operations | Update embeddings as data changes |
| Scaling | Handle millions to billions of vectors |

Vector databases use ANN (Approximate Nearest Neighbor) algorithms, allowing us to search through millions or billions of documents efficiently. ANN reduces search time complexity from $O(N)$ (linear search) to $O(\log N)$. While linear search offers 100% accuracy, ANN provides 95-99% accuracy in a fraction of the time. This massive increase in speed is why modern vector stores rely on ANN.

### Choices of Vector Stores

| Database | Type | Best For | Pricing Model |
|----------|------|----------|---------------|
| **Pinecone** | Managed cloud | Easy start, scale, managed SLAs | Per vector-hour |
| **Qdrant** | Open source / Cloud | Self-hosted control, lightning-fast performance | Per GB (cloud) or free |
| **Weaviate** | Open source / Cloud | Native hybrid search, multimodal | Per dimension-hour |
| **Milvus** | Open source / Cloud | Distributed scale (50M+ vectors) | Free (self-host) or Cloud |
| **Chroma** | Open source | Prototyping, local development | Free |

### What I Used and Why

I chose **Qdrant** as the vector database because it is open-source, fast, and extremely easy to self-host. In this project, Qdrant runs inside a Docker container on port 6333 with a mounted volume for persistent storage.

## Step 5: Retrieval (Hybrid Search)

Retrieval is the core—and often the most complex—part of a RAG system. It is the process of selecting the right chunks from the vector store based on a user's query.

There are two primary architectures for retrieval:

### 1. Simple Dense Search
- Takes the query, encodes it into a dense embedding using the same model used for the chunks, and computes semantic similarity (e.g., Cosine Similarity).
- Retrieves the top-K most relevant chunks.
- Relies entirely on dense embeddings.

### 2. Hybrid Search
- Combines dense (semantic) and sparse (keyword) retrieval to get the best of both worlds. 
- It is the baseline for production RAG systems. Modern databases like Qdrant and Weaviate offer native hybrid pipelines out of the box.

To understand Hybrid RAG, you need to understand Dense and Sparse retrieval:

#### Dense Retrieval
Uses neural embeddings to match meaning.

**Strengths:**
- Understands paraphrasing and synonyms.
- Captures conceptual similarity.
- Works across languages (with multilingual models).

**Weaknesses:**
- May miss exact keyword matches.
- Struggles with specific entities, serial numbers, and acronyms.

#### Sparse Retrieval
Uses term frequency and statistics (e.g., BM25, TF-IDF).

**Strengths:**
- Excellent for exact keyword matches.
- Handles rare terms, codes, and specific entities perfectly.
- Fast and requires no neural training.

**Weaknesses:**
- Misses semantic similarity and contextual meaning.
- Lacks synonym understanding.
- Sensitive to vocabulary mismatch (e.g., "car" vs. "automobile").

### What I Used and Why

I implemented **Hybrid Search**. It provides the best of both worlds, combining the deep semantic understanding of dense retrieval with the exact keyword matching of sparse retrieval. This makes it the premier choice for a production-level RAG system.

### Parallel Retrieval with Fusion

```text
                    +------------------+
                    |      Query       |
                    +--------+---------+
                             |
              +--------------+--------------+
              v                             v
    +-------------------+         +-------------------+
    |  Dense Retrieval  |         |  Sparse Retrieval |
    |   (Vector DB)     |         |    (BM25/ES)      |
    +---------+---------+         +---------+---------+
              |                             |
              +--------------+--------------+
                             v
                    +-------------------+
                    |      Fusion       |
                    |    (weighted)     |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |  Final Results    |
                    +-------------------+
```

### Fusion Methods

Fusion is the algorithm used to combine the ranked lists from two different search algorithms.

#### Reciprocal Rank Fusion (RRF)
RRF is a popular standard for combining results. It completely ignores the raw *scores* (which are often incomparable across different algorithms) and looks purely at the *rank* of the documents.

**Properties:**
- Position-based; ignores raw numerical scores.
- Robust to score scale differences, preventing one algorithm from dominating.
- Simple to implement with no complex tuning required.

#### Weighted Score Fusion
Combines and averages the normalized raw scores from both retrieval methods.

**Properties:**
- Uses actual scores, retaining more information than pure rank.
- Requires careful score normalization.
- Requires tuning an "alpha" parameter to control the balance between dense and sparse results.

### What is Used in This Project and Why?

I experimented with both RRF and Weighted Score Fusion.

- RRF achieved an MRR of **0.79**.
- Weighted Fusion required fine-tuning the alpha value. I tested alpha values from 0.1 to 0.9 (where 0.1 means 10% weight to dense and 90% to sparse).
- A dense weight of **0.4** provided the best overall performance, achieving a superior MRR of **0.8223**.

## Step 6: Reranking

There are two primary methods for reranking retrieved documents:
1. Cross-Encoders
2. LLM-as-a-Judge

### What I Used and Why

#### Cross-Encoders
I benchmarked the system with and without a Cross-Encoder. The Cross-Encoder improved the retrieval accuracy of the absolute top chunk by only ~3%, but it increased latency by 10x. Because of this harsh trade-off, I decided against using a Cross-Encoder.

#### LLM Relevancy Filter
Instead, I integrated a **Relevancy Layer** directly into the AI architecture. An LLM evaluates the retrieved documents to filter out irrelevant chunks before passing the final context to the generation node. This acts as an intelligent, context-aware filter and reranker without the massive latency penalty of a Cross-Encoder.

## Step 7: Evaluation

### Dataset Preparation
To test retrieval performance, we need a "Golden Ground Truth" dataset containing pairs of Queries and the IDs of the most relevant documents. 
**How to prepare it:** We reversed the process. We took random chunks from our dataset and used an LLM to generate highly specific questions that could only be answered by that exact chunk.

### Metrics Explained
- **Recall@K:** Measures whether the correct chunk appeared anywhere in the top K retrieved documents. For example, Recall@3 tells us if the relevant document is in the top 3.
- **MRR (Mean Reciprocal Rank):** Evaluates how close the correct chunk is to the #1 spot on average.
- **Precision@K:** Out of the top K chunks retrieved, how many were actually relevant?

## The Results

### Dense-Only Retrieval
- **Recall@1:** 0.5958
- **Recall@2:** 0.7246
- **Recall@3:** 0.7904
- **Recall@4:** 0.8443
- **Recall@5:** 0.8653
- **MRR:** 0.6998

### Hybrid Search Results (For Top 10)
*Latency: 1 minute 15 seconds for 334 Queries*
- **Recall@1:** 0.7156
- **Recall@2:** 0.8263
- **Recall@3:** 0.8922
- **Recall@4:** 0.9192
- **Recall@5:** 0.9341
- **MRR:** 0.8085

### Cross-Encoder Results
*Latency: 161 minutes for 10 chunks retrieved per query*
- **Recall@1:** 0.7455
- **Recall@2:** 0.8653
- **Recall@3:** 0.9012
- **Recall@4:** 0.9281
- **Recall@5:** 0.9491
- **MRR:** 0.8283

### ColBERT Results
*Latency: 1 minute overall*
- **Recall@1:** 0.7096
- **Recall@2:** 0.8473
- **Recall@3:** 0.8862
- **Recall@4:** 0.9222
- **Recall@5:** 0.9311
- **MRR:** 0.8022

### Key Observations
- Cross-Encoders are only worth the computational cost if absolute Precision@1 is your only priority.
- For finding relevant chunks in the Top 3 to Top 5, Hybrid Search provides excellent results with drastically lower latency.
- Dense-only retrieval should generally be avoided in production. 
- With clever chunking strategies to bump highly relevant chunks to the top, Hybrid Search can perform nearly as well as a Cross-Encoder at a fraction of the cost.

### Using Weighted Fusion:
After fine-tuning the alpha value between 0.1 and 0.9, the results were clear:

🏆 **Best Configuration:** Highest MRR (0.819) was achieved with a Dense Weight of **0.4** and a Sparse Weight of **0.6**.

This means that if we look at the top 3 retrieved chunks, there is an over **90% probability** that we have the exact chunk needed to answer the user's query!