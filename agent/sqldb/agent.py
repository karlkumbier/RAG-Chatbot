from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage

from agent.sqldb import nodes
from typing_extensions import TypedDict
from typing import Sequence, Annotated, Dict

import operator
import pandas as pd

class SQLDBAgent(StateGraph):
  
  def __init__(self, name="sqldb"):
    """ Initializes graph state """    
    state = {
      "question": str,
      f"{name}_messages": Annotated[Sequence[BaseMessage], operator.add],
      f"{name}_ntry": int,
      f"{name}_dialect": str,
      f"{name}_schema": str,
      f"{name}_query": str,
      f"{name}_df": pd.DataFrame,
      f"{name}_df_summary": str
    }

    state = TypedDict("AgentState", state)
    self.agent = self.__construct_graph__(state)
    self.name = name
    self.state = None
    
  def __construct_graph__(self, state): 
    """ Initializes logic flow of agent graph """
    workflow = StateGraph(state)
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
      return self.state.get(f"{self.name}_{key}")


if __name__ == "__main__":
  from agent.sqldb.agent import SQLDBAgent
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
  
  name = sqldb_agent.name
  print(results[f"{name}_query"])
  print(results[f"{name}_df_summary"])
  print(results[f"{name}_df"])
  
  # Test 2:
  question = """
    Generate a table of differential expression results that includes data on 
    all available cell lines. Include all available columns. Use only the 
    baseline context for each cell line and the standard of care versus no drug 
    contrast. Return all rows from this table.  
  """
  
  results = sqldb_agent.invoke({"question": question}, config)
  print(results[f"{name}_query"])
  print(results[f"{name}_df_summary"])
  print(results[f"{name}_df"])