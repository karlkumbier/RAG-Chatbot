from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from typing import List, Dict

from agent.base_agent.agent import BaseAgent
from agent.rag import nodes
from agent.models import gpt4, retriever
LLM = gpt4

###############################################################################
# Initialize agent graph 
###############################################################################
class RAGAgentState(TypedDict):
  question : str # user question / request
  results: Dict # output response generated by llm
  documents : List[str] # retrieved docs

class RAGAgent(BaseAgent):
  
  def __init__(self, name='rag'):
    super().__init__()
    self.agent = self.__build_graph__()
    self.name = name
    
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

if __name__ == "__main__":
  from agent.rag.agent import *
  rag_agent = RAGAgent()
  
  question = """What are cancer persisters?"""
  config = {"name": "rag", "llm": LLM, "retriever": retriever}
  rag_agent.__set_state__({"question":question}, config)
  rag_agent.get("result")
  