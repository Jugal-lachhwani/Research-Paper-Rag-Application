# pyrefly: ignore [missing-import]
import operator
from typing import Annotated, Any, Dict, List, TypedDict

class GraphState(TypedDict):

    messages: Annotated[List[Dict[str, str]], operator.add]
    query: str
    extraction_needed: str
    retrieved_docs: List[Dict[str, Any]]
    relevant_docs: List[Dict[str, Any]]
