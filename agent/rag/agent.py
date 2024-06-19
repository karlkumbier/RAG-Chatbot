from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from typing import List, Sequence, Annotated, Dict
from langchain_core.messages import BaseMessage

from agent.rag import nodes
from agent.models import gpt4, retriever
import operator

LLM = gpt4

###############################################################################
# Initialize agent graph 
###############################################################################
class RAGAgentState(TypedDict):
  question : str
  response: str
  messages : Annotated[Sequence[BaseMessage], operator.add]
  context : str
  documents : List[str]

class RAGAgent:
  
  def __init__(self, name='rag'):
    self.agent = self.__build_graph__()
    self.name = name
    self.state = None
    
  def __build_graph__(self):
    workflow = StateGraph(RAGAgentState)
    workflow.add_node("initializer", nodes.initialize)
    workflow.add_node("retriever", nodes.retrieve)
    workflow.add_node("generator", nodes.generate)

    workflow.set_entry_point("initializer")
    workflow.add_edge("initializer", "retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "__end__")
    return workflow.compile()

    
  def __print__(self):
    if self.agent is None:
      raise Exception("Agent graph not built")
    
    self.agent.get_graph().print_ascii()
    
  def invoke(self, state: Dict, config: Dict):
    return self.agent.invoke(state, config)
  
  def __clear_state__(self):
    self.state = None
    
  def get(self, key: str):
    if self.state is None:
      return None
    else:
      return self.state.get(key)

if __name__ == "__main__":
  from agent.rag.agent import *
  rag_agent = RAGAgent()
  question = """What are cancer persisters?"""
  config = {"name": "rag", "llm": LLM, "retriever": retriever}
  result = rag_agent.invoke({"question":question}, config)
  print(result.get("messages")[-1].content)