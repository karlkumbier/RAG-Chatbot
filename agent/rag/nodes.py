
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from agent.rag.utils import format_docs
from agent.rag.prompts import RAG_PROMPT
from agent.models import gpt4
from typing import Dict

LLM = gpt4

# Set agent initialer and router
def initialize(state: Dict, config: Dict) -> Dict:
  """ Initialize messages"""
  if config.get("retriever") is None:
    raise Exception("No retriever provided for RAG")
  
  if state.get("messages") is None:
    state["messages"] = []
    
  return state


def retrieve(state: Dict, config: Dict) -> Dict:
    """Retrieve documents from vectorstore."""
    retriever = config.get("retriever")

    # Retrieval
    state["documents"] = retriever.invoke(state["question"])
    state["context"]= format_docs(state["documents"])
    return state


def generate(state: Dict, config: Dict) -> Dict:
  """ Generate response"""
  llm = config.get("llm", LLM)
  
  prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
  ])
  
  chain = prompt | llm | StrOutputParser()
  state["response"] = chain.invoke(state)

  return state