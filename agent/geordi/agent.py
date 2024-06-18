from agent.models import gpt4
from agent.geordi.prompts import *
from langgraph.graph import StateGraph
from langchain.utilities.sql_database import SQLDatabase

# Initialize SQLDB base agent
from agent.sqldb.agent import SQLDBAgent
from agent.chart.agent import ChartAgent
from agent.geordi import nodes

from langchain_core.messages import BaseMessage
from typing import Sequence, Annotated, Dict
from typing_extensions import TypedDict
import operator

LLM = gpt4

class GeordiAgentState(TypedDict):
  question: str # user question
  messages: Annotated[Sequence[BaseMessage], operator.add] # chat history
  sqldb_agent: SQLDBAgent # database agent
  chart_agent: ChartAgent # figure generating agent
  ntry: int # number of debug attemps

class GeordiAgent:
  def __init__(self, name="geordi"):
    """ Initializes graph state """    
    self.agent = self.__construct_graph__()
    self.name = name
    self.state = None
    
  def __construct_graph__(self): 
    """ Initializes logic flow of agent graph """
    workflow = StateGraph(GeordiAgentState)
    workflow.add_node("initializer", nodes.initialize)
    workflow.add_node("data_loader", nodes.load_data)
    workflow.add_node("router_explainer", nodes.explain_router)
    workflow.add_node("figure_generator", nodes.make_figure)
    workflow.add_node("analysis_runner", nodes.run_analysis)

    workflow.set_entry_point("initializer")
    workflow.add_edge("initializer", "data_loader")
    workflow.add_edge("router_explainer", "data_loader")
    workflow.add_edge("figure_generator", "__end__")
    workflow.add_edge("analysis_runner", "__end__")

    workflow.add_conditional_edges(
      "data_loader",
      nodes.route,
      {
        "explain": "router_explainer", 
        "figure": "figure_generator", 
        "analysis": "analysis_runner",
        "end": "__end__"
      }
    )

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
  from agent.geordi.agent import *
  username = "picard"
  password = "persisters"
  port = "5432"
  host = "localhost"
  dbname = "persisters"
  pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"
  db = SQLDatabase.from_uri(pg_uri)
  
  geordi = GeordiAgent()
  geordi.__print__()
  config = {"verbose": True, "db": db, "llm": LLM, "name": "geordi"}

  # Test 1 
  question = """Generate a figure of differential expression p-value versus 
    log2 fold change for PC9 cell lines. The differential expression analysis 
    should contrast SOC with no drug. Use data from the hypoxia vs. normoxia 
    screen. 
  """

  result = geordi.invoke({"question": question}, config=config)
  print(result["chart_agent"].get("fig"))
  result["chart_agent"].state
  
  
  result.keys()
  ##############################################################################
  # If data are not relevant, describe problem, retry query
  ##############################################################################

  ##############################################################################
  # Once data are relevant...
  ##############################################################################

  ##############################################################################
  # Determine task to be performed - task should be in state, to be read by 
  # supervising agents
  ##############################################################################

  ##############################################################################
  # Carry out task
  # 1. Update figure agent according to state / table summary / new changes
  # 2. Add node for `analysis` to return message: methods not implemented yet
  # 3. Modify state (e.g., by adding fig). Add state message summarizing the fig.
  ##############################################################################

  ##############################################################################
  # Assess task completeness - chat history / single message node
  ##############################################################################

  ##############################################################################
  # If task not complete, describe problem, route description to one of database # loader or figure generator and rerun cycle
  ##############################################################################

