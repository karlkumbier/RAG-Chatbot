from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from typing import List, Dict
from agent.models import gpt4

RAG_LLM = gpt4

def retriever_node_(state: Dict, retriever: BaseRetriever) -> Dict:
    """Retrieve documents from vectorstore."""
    question = state.get("question")

    # Retrieval
    state["documents"] = retriever.invoke(question)
    state["context"]= format_docs(state["documents"])
    return state

def format_docs(docs: List[Document]) -> str:
    """Convert list of Documents to a formated string for LLM ingestion"""
    formatted = [
        f"Article Reference: {doc.metadata['reference']}\n\n \n\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)