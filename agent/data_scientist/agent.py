from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.utilities.sql_database import SQLDatabase

from agent.base_agent.agent import BaseAgent
from agent.data_scientist import nodes
from agent.data_scientist.prompts import *
from agent.models import gpt4

from langchain_core.messages import BaseMessage
from typing import Sequence, Annotated, Dict
from typing_extensions import TypedDict

LLM = gpt4

class DataScientistState(TypedDict):
  question: str # user question
  messages: Annotated[Sequence[BaseMessage], add_messages] # chat history
  sqldb_state: Dict # database agent state
  chart_state: Dict # chart agent state
  results: Dict # agent output
  ntry: int # number of debug attemps

class DataScientistAgent(BaseAgent):
  def __init__(self, name="geordi"):
    super().__init__() 
    self.agent = self.__build_graph__()
    self.name = name
    
  def __build_graph__(self): 
    """ Initializes logic flow of agent graph """
    workflow = StateGraph(DataScientistState)
    workflow.add_node("initializer", nodes.initialize)
    workflow.add_node("data_loader", nodes.load_data)
    workflow.add_node("router_explainer", nodes.explain_router)
    workflow.add_node("figure_generator", nodes.make_figure)
    workflow.add_node("analysis_runner", nodes.run_analysis)
    workflow.add_node("table_generator", nodes.make_table)

    workflow.set_entry_point("initializer")
    workflow.add_edge("initializer", "data_loader")
    workflow.add_edge("router_explainer", "data_loader")
    workflow.add_edge("figure_generator", "__end__")
    workflow.add_edge("analysis_runner", "__end__")
    workflow.add_edge("table_generator", "__end__")
    
    workflow.add_conditional_edges(
      "data_loader",
      nodes.route,
      {
        "explain": "router_explainer", 
        "figure": "figure_generator", 
        "analysis": "analysis_runner",
        "table": "table_generator"
      }
    )

    return workflow.compile()

if __name__ == "__main__":
  from agent.data_scientist.agent import *
  username = "picard"
  password = "persisters"
  port = "5432"
  host = "localhost"
  dbname = "persisters"
  pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"
  db = SQLDatabase.from_uri(pg_uri)
  
  geordi = DataScientistAgent()
  geordi.__print__()
  config = {"verbose": True, "db": db, "llm": LLM, "name": "geordi"}

  # Test 1 
  question = """Generate a figure of differential expression p-value versus 
    log2 fold change for PC9 cell lines. The differential expression analysis 
    should contrast SOC with no drug. Use data from the hypoxia vs. normoxia 
    screen. 
  """

  result = geordi.invoke({"question": question}, config=config)
  print(result["results"])
  
  # Test 2
  question = """Combine differential expression results from the January 2023 
  and April 2023 screens for all cell lines. The differential expression 
  analysis should contrast SOC with no drug. Filter to rows from the baseline 
  context. Using these data, generate a figure of differential expression 
  p-value versus log2 fold change by cell line. Your figure should have one subplot per cell line.
  """

  result = geordi.invoke({"question": question}, config=config)
  print(result["sqldb_agent"].state["query"]) 
  print(result["chart_agent"].state["fig_summary"]) 
  
  
  
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

