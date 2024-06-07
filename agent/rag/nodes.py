from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from typing import List
from agent.rag.prompt import RAG_PROMPT
from agent.models import gpt4

RAG_LLM = gpt4

def retrieve(state):
    """Retrieve documents from vectorstore."""
    question = state.get("question")

    # Retrieval
    state["documents"]= state.get("retriever").invoke(question)
    return state

def generate(state):
    """Generate answer using RAG on retrieved documents."""
    question = state.get("question")
    docs = format_docs(state["documents"])
    
    prompt = PromptTemplate.from_template(template=RAG_PROMPT)
    chain = prompt | RAG_LLM | StrOutputParser()
    
    # RAG generation
    response = chain.invoke({"context": docs, "question": question})
    state["rag_response"] = response
    return state

def format_docs(docs: List[Document]) -> str:
    """Convert list of Documents to a formated string for LLM ingestion"""
    formatted = [
        f"Article Reference: {doc.metadata['reference']}\n\n \n\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)