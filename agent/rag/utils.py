from langchain_core.documents import Document
from typing import List

def format_docs(docs: List[Document]) -> str:
    """Convert list of Documents to a formated string for LLM ingestion"""
    formatted = [
        f"Article Reference: {doc.metadata['reference']}\n\n \n\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)