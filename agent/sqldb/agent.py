from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from agent.sqldb import nodes
from typing_extensions import TypedDict
from typing import Sequence, Annotated, Dict
import pandas as pd

class SQLDBAgentState(TypedDict):
  question: str # user question
  messages: Annotated[Sequence[BaseMessage], add_messages] # chat history
  ntry: int # number of attempts
  dialect: str # SQL dialect
  schema: str # comments on tables + columns
  query: str # SQL query
  df: pd.DataFrame # loaded data frame
  df_summary: str # description of laoded data frame


class SQLDBAgent(StateGraph):
  
  def __init__(self, name="sqldb"):
    """ Initializes graph state """    
    self.agent = self.__construct_graph__()
    self.name = name
    self.state = None
    
  def __construct_graph__(self): 
    """ Initializes logic flow of agent graph """
    workflow = StateGraph(SQLDBAgentState)
    workflow.add_node("initializer", nodes.initialize)
    workflow.add_node("schema_initializer", nodes.set_schema)
    workflow.add_node("query_generator", nodes.generate_query)
    workflow.add_node("query_runner", nodes.run_query)
    workflow.add_node("query_debugger", nodes.debug_query)
    workflow.add_node("table_summarizer", nodes.summarize_table)

    workflow.set_entry_point("initializer")
    workflow.add_edge("initializer", "schema_initializer")
    workflow.add_edge("schema_initializer", "query_generator")
    workflow.add_edge("query_generator", "query_runner")
    workflow.add_edge("query_debugger", "query_runner")
    workflow.add_edge("table_summarizer", "__end__")

    workflow.add_conditional_edges(
      "query_runner",
      nodes.route,
      {
        "debug": "query_debugger", 
        "summarize": "table_summarizer", 
        "__end__": "__end__"
      },
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
  #from agent.sqldb.agent import * 
  from agent.models import gpt4
  from langchain.utilities.sql_database import SQLDatabase

  # Initialize connection to database 
  username = "picard"
  password = "persisters"
  port = "5432"
  host = "localhost"
  dbname = "persisters"
  pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"
  db = SQLDatabase.from_uri(pg_uri)
  
  # Initialize agent
  sqldb_agent = SQLDBAgent()
  sqldb_agent.__print__()


  # Test 1:
  question = """
    Get a tables of log2 fold change, p-value, and gene name from the hypoxia 
    vs. normoxia screen. Filter these to PC9 cell lines in the baseline
    context. Limit results to 20 samples
  """
  
  config = {"db": db, "llm": gpt4, "name": sqldb_agent.name, "verbose": True}
  results = sqldb_agent.invoke({"question": question}, config)
  
  print(results["query"])
  print(results["df_summary"])
  print(results["df"])
  
  # Test 2:
  question = """
    Generate a table of differential expression results that includes data on 
    all available cell lines. Include all available columns. Use only the 
    baseline context for each cell line and the standard of care versus no drug 
    contrast. Return all rows from this table.  
  """
  
  results = sqldb_agent.invoke({"question": question}, config)
  print(results["query"])
  print(results["df_summary"])
  print(results["df"])