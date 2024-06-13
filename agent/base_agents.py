from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseLanguageModel
from typing import Dict

def agent_node(state: Dict, agent: Runnable, name: str) -> Dict:
  """ Wrapper function for message handling langraph node """
  result = agent.invoke(state)
  result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
  
  # Convert agent output into a format suitable to append to the global state
  state["messages"] = [result]
  state["sender"] = name
  
  if state.get("ntry") is not None: 
    state["ntry"] += 1
  
  return state

def chat_agent(llm: BaseLanguageModel, base_prompt: str) -> str:
  """ Wrapper function for chat agent"""
  prompt = ChatPromptTemplate.from_messages([
    ("system", base_prompt),
    MessagesPlaceholder(variable_name="messages"),
  ])   
  
  return prompt | llm
