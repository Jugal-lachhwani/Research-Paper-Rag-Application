from langchain_core.prompts import PromptTemplate


EXTRACTION_PROMPT = PromptTemplate.from_template(
    """You are an intelligent routing assistant.
Your task is to determine whether the following user query requires searching an external knowledge base for specific factual information, research papers, or deep context.

If the query requires factual data, research papers, or specific knowledge, respond with exactly "Yes".
If the query is a simple greeting, general conversation, or something you can answer without external facts, respond with exactly "No".

User Query: {query}
Answer (Yes/No):"""
)

RELEVANCE_PROMPT = PromptTemplate.from_template(
    """You are a strict evaluator. Your task is to determine if the retrieved document contains relevant information to answer the user query.
Ignore formatting, but focus on the semantic meaning.

If the document contains useful information to help answer the query, respond with exactly "Yes".
If the document is completely irrelevant or off-topic, respond with exactly "No".

Query: {query}

Document Context:
{document_content}

Answer (Yes/No):"""
)

SUMMARY_PROMPT = PromptTemplate.from_template(
    """You are a helpful summarization assistant.
Please summarize the following context concisely, extracting the core concepts, entities, and facts. Keep it under 250 words.

Context:
{context}

Summary:"""
)

GENERATE_WITH_CONTEXT_PROMPT = PromptTemplate.from_template(
    """You are an expert research assistant. Answer the user's question based strictly on the provided context below.
If the context does not contain enough information to answer the question, state that you do not have enough information.

Context:
{context}

Question: {query}
Answer:"""
)

GENERATE_DIRECT_PROMPT = PromptTemplate.from_template(
    """You are a helpful and intelligent assistant. Please answer the user's question directly and concisely.

Question: {query}
Answer:"""
)

RAG_PROMPT = GENERATE_WITH_CONTEXT_PROMPT
Generation_prompt = GENERATE_DIRECT_PROMPT

