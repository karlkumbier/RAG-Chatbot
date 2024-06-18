from langchain.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage

from agent.sqldb import nodes
from typing_extensions import TypedDict
from typing import Sequence, Annotated, Dict
from agent.models import gpt4

import operator
import pandas as pd

# Initialize base parameters for agent
LLM = gpt4

# Initialize connection to database 
username = "picard"
password = "persisters"
port = "5432"
host = "localhost"
db = "persisters"
pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db}"
DB = SQLDatabase.from_uri(pg_uri)


class SQLDBAgent:
  
  def __init__(self, name="sqldb"):
    
    state = {
      f"{name}_question": str,
      f"{name}_messages": Annotated[Sequence[BaseMessage], operator.add],
      f"{name}_ntry": int,
      f"{name}_dialect": str,
      f"{name}_schmea": str,
      f"{name}_query": str,
      f"{name}_df": pd.DataFrame,
      f"{name}_df_summary": str
    }

    self.name = "name" 
    self.AgentState = TypedDict("AgentState", state)
    self.__construct_graph__()

  def __construct_graph__(self): 
    workflow = StateGraph(self.AgentState)
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
        "describe": "table_summarizer", 
        "__end__": "__end__"
      },
    )
    
    self.agent = workflow.compile()
    
  def __print__(self):
    if self.agent is None:
      print("Graph not initialized")
    else:
      self.agent.get_graph().print_ascii()

  def invoke(self, state: Dict, config: Dict):
    return self.agent.invoke(state=state, config=config)
    
if __name__ == "__main__":
  sqldb_agent = SQLDBAgent(db=db, name="sqldb", llm=LLM)
  sqldb_agent.__print__()
    
  # Test 1:
  question = """
    Get a tables of log2 fold change, p-value, and gene name from the hypoxia screen. Filter these to PC9 cell lines. Limit results to 20 samples
  """

  config = {"db": db, "llm": LLM, "name": sqldb_agent.name}
  results = sqldb_agent.invoke({"question": question}, config=config)
  results.keys()
  
  print(results["db_query"])
  print(results["df_summary"])
  print(results["df"])
  
  # Test 2:
  question = """
    Generate a table of differential expression results that includes data on all available cell lines. Include all available columns. Use only the baseline context for each cell line and the standard of care versus no drug contrast. Return all rows from this table.  
  """
  
  results = sqldb_agent.invoke({"question": question})
  print(results["db_query"])
  print(results["df"])
  