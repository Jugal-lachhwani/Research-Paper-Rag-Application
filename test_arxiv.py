import os
import arxiv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class ArxivQuery(BaseModel):
    query: str = Field(description="The exact search keywords extracted from the user query. Enclose specific phrases in double quotes (e.g. 'all:\"Attention is All you need\"' or '\"Attention is All you need\"').")
    newest_preference: str = Field(description="If the user explicitly asks for the latest or newest papers, answer 'Yes', otherwise answer 'No'.")

def search_arxiv_smart(user_query: str):
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    structured_llm = llm.with_structured_output(ArxivQuery)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a research assistant. Extract the core search string from the user's request. Also determine if the user explicitly prefers the newest/latest papers ('Yes' or 'No')."),
        ("human", "{query}")
    ])
    
    chain = prompt | structured_llm
    result = chain.invoke({"query": user_query})
    
    if result.newest_preference.lower() == "yes":
        sort_criterion = arxiv.SortCriterion.SubmittedDate
    else:
        sort_criterion = arxiv.SortCriterion.Relevance
        
    search = arxiv.Search(
        query=result.query,
        max_results=5,
        sort_by=sort_criterion,
        sort_order=arxiv.SortOrder.Descending
    )
    
    print(f"User Query: '{user_query}'")
    print(f"LLM Extracted Keywords: {result.query}")
    print(f"Newest Preference: {result.newest_preference} (Using {sort_criterion})")
    print("=" * 50)
    
    client = arxiv.Client()
    results = list(client.results(search))
    
    if not results:
        print("No papers found.")
        return
        
    for i, paper in enumerate(results, 1):
        print(f"[{i}] {paper.title}")
    print("\n")

print("=== QUERY 1 (Normal Relevance) ===")
search_arxiv_smart('Attention is All you need')

print("=== QUERY 2 (Newest Preference) ===")
search_arxiv_smart('Find the newest papers on "Attention is All you need"')
