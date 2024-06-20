from agent.geordi.agent import GeordiAgent
from agent.rag.agent import RAGAgent
from agent.models import gpt4
from agent.picard.prompts import *
from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
  ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
)

LLM = gpt4

# TODO: handle complex task breakdown - decide task and worker to perform task
# TODO: worker descriptions passed in with worker agents / names

def initialize(state: Dict, config: Dict) -> Dict:
    
  # Define set of workers picard will have access to
  state["workers"] = {"geordi":GeordiAgent(), "rag": RAGAgent()}
  
  # Check configs to ensure alignment with workers
  if state.get("messages") is None:
    state["messages"] = []
    
  if state.get("ntry") is None:
    state["ntry"] = 0
    
  return state


def distil_task(state: Dict, config: Dict) -> Dict:
  """ Determine requested task based on chat history """
  llm = config.get("llm", LLM)
  messages = state.get("messages")  
  
  prompt = ChatPromptTemplate.from_messages([
    ("system", DISTILL_TASK_PROMPT),
    MessagesPlaceholder(variable_name="messages")
  ])

  chain = prompt | llm | StrOutputParser()
  state["question"] = chain.invoke({"messages": messages})
  
  return state


def assign_task(state: Dict, config: Dict) -> Dict:
  """ Assign task to one of the sub-agents """
  llm = config.get("llm", LLM)
  prompt = PromptTemplate.from_template(ASSIGN_TASK_PROMPT)
  chain = prompt | llm | StrOutputParser()
  
  result = chain.invoke(state)
  state["active_worker"] = result
  
  return(state)


def run_task(state: Dict, config: Dict) -> Dict:
  """ Performs task based on assigned worker"""
  active_worker = state["active_worker"]
  worker = state["workers"][active_worker]
  state["workers"][active_worker] = worker.invoke(state, config)
  
  return state


def output(state: Dict, config: Dict) -> Dict:
  """ Returns desired entity, description"""
  active_worker = state["active_worker"]
  state["output"] = {
    "output":state["workers"][active_worker].get("output")
  }
  ...
  