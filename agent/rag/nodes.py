
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
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

  state["results"] = {}    
  return state


def retrieve(state: Dict, config: Dict) -> Dict:
    """Retrieve documents from vectorstore."""
    retriever = config.get("retriever")
    state["documents"] = retriever.invoke(state["question"])
    return state


def generate(state: Dict, config: Dict) -> Dict:
  """ Generate response"""
  llm = config.get("llm", LLM) 
  prompt = PromptTemplate.from_template(RAG_PROMPT)
  chain = prompt | llm | StrOutputParser()
  
  state["context"]= format_docs(state["documents"])
  state["results"]["response"] = chain.invoke(state)
  state["results"]["type"] = "response"
  return state