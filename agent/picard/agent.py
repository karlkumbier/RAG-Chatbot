from agent.models import gpt4
from langgraph.graph import StateGraph
from typing import Dict
from langchain.utilities.sql_database import SQLDatabase
from agent.picard import nodes
from langchain_core.messages import BaseMessage
from typing import Sequence, Annotated, Dict
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage
from agent.models import retriever
from langgraph.graph.message import add_messages

# TODO: chanage from operator.add to add_messages in other agents
# TODO: output nodes on agents - component of state that is desired output of 
#   request. Of arbitrary type.
# TODO: data table node on geordi

class PicardState(TypedDict):
  question: str # user question
  messages: Annotated[Sequence[BaseMessage], add_messages] # chat history
  workers: Dict # database agent
  active_worker: str
  ntry: int # number of debug attemps

class PicardAgent:
  def __init__(self, name="picard"):
    """ Initializes graph state """    
    self.agent = self.__construct_graph__()
    self.name = name
    self.state = None
    
  def __construct_graph__(self): 
    """ Initializes logic flow of agent graph """
    workflow = StateGraph(PicardState)
    workflow.add_node("initializer", nodes.initialize)
    workflow.add_node("task_distiller", nodes.distil_task)
    workflow.add_node("task_assigner", nodes.assign_task)
    workflow.add_node("task_runner", nodes.run_task)
    
    workflow.set_entry_point("initializer")
    workflow.add_edge("initializer", "task_distiller")
    workflow.add_edge("task_distiller", "task_assigner")
    workflow.add_edge("task_assigner", "task_runner")
    workflow.add_edge("task_runner", "__end__")

    return workflow.compile()
  
  def __print__(self):
    if self.agent is None:
      print("Graph not initialized")
    else:
      self.agent.get_graph().print_ascii()

  def invoke(self, state: Dict, config: Dict):
    return self.agent.invoke(state, config=config)
    
  def __clear_state__(self):
    self.state = None
    
  def get(self, key: str):
    if self.state is None:
      return None
    else:
      return self.state.get(key)

if __name__ == "__main__":
  from agent.picard.agent import *
  
  username = "picard"
  password = "persisters"
  port = "5432"
  host = "localhost"
  dbname = "persisters"
  pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"
  db = SQLDatabase.from_uri(pg_uri)
  
  picard = PicardAgent()
  picard.__print__()
  
  config = {"verbose": True, "db": db, "llm": gpt4, "retriever": retriever}
  question = "What are cancer persisters?"
  messages = [HumanMessage(question)]
  
  result = picard.invoke({"messages": messages}, config)
  result["messages"]
  result["question"]
  result["active_worker"]