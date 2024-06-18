from langchain.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage

from typing_extensions import TypedDict
from typing import Sequence, Annotated, Literal, Dict
from agent.models import cold_gpt35, gpt4
from agent.base_agents import chat_agent, agent_node
from agent.sqldb.prompts import SQL_QUERY_PROMPT, SQL_DEBUGGER_PROMPT

from agent.sqldb.utils import (
  set_schema_node_, run_query_node_, summarize_table_node_
)

import functools
import operator
import pandas as pd

# Initialize base parameters for agent
LLM = gpt4
NTRY = 10

# Initialize connection to database 
username = "picard"
password = "persisters"
port = "5432"
host = "localhost"
db = "persisters"

pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db}"
db = SQLDatabase.from_uri(pg_uri)

###############################################################################
# Initialize agent graph 
###############################################################################
# Define agent state
class DBAgentState(TypedDict):
  question: str # user question
  messages: Annotated[Sequence[BaseMessage], operator.add] # chat history
  db_query: str # last SQL query
  df: pd.DataFrame # dataframe from query
  df_summary: str # description of dataframe
  db_schema: str # summary of SQL tables
  dialect: str # SQL dialect
  sender: str # DEPRICATED; last message node
  ntry: int # number of debug attemps


# Define graph node functions
def initializer_node(state: Dict) -> Dict:
  """ Initialize tools and evaluate DB schema"""
  if state.get("messages") is None:
    state["messages"] = []
    
  if state.get("ntry") is None:
    state["ntry"] = 0
    
  return state
  
def router(state: Dict) -> Literal["debug", "__end__"]:
  """ Routes agent action to either debug or program end """ 
  if state.get("df") is None:
    return "__end__" if state["ntry"] > NTRY else "debug"
  else:
    return "describe"

generate_query_node = functools.partial(
  agent_node, 
  agent=chat_agent(LLM, SQL_QUERY_PROMPT), 
  name="query_generator"
)

debug_query_node = functools.partial(
  agent_node,
  agent=chat_agent(LLM, SQL_DEBUGGER_PROMPT),
  name="query_debugger"
)

summarize_table_node = functools.partial(
  summarize_table_node_,
  llm=LLM,
  name="table_summarizer"
)

run_query_node = functools.partial(
  run_query_node_,
  db=db,
  name="query_runner"  
)

set_schema_node = functools.partial(
  set_schema_node_,
  llm=LLM,
  db=db
)

# Construct agent graph
workflow = StateGraph(DBAgentState)
workflow.add_node("initializer", initializer_node)
workflow.add_node("schema_initializer", set_schema_node)
workflow.add_node("query_generator", generate_query_node)
workflow.add_node("query_runner", run_query_node)
workflow.add_node("query_debugger", debug_query_node)
workflow.add_node("table_summarizer", summarize_table_node)

workflow.set_entry_point("initializer")
workflow.add_edge("initializer", "schema_initializer")
workflow.add_edge("schema_initializer", "query_generator")
workflow.add_edge("query_generator", "query_runner")
workflow.add_edge("query_debugger", "query_runner")
workflow.add_edge("table_summarizer", "__end__")

workflow.add_conditional_edges(
  "query_runner",
  router,
  {
    "debug": "query_debugger", 
    "describe": "table_summarizer", 
    "__end__": "__end__"
  },
)

sqldb_agent = workflow.compile()
sqldb_agent.get_graph().print_ascii()

if __name__ == "__main__":

  # Test 1:
  question = """
    Get a tables of log2 fold change, p-value, and gene name from the hypoxia screen. Filter these to PC9 cell lines. Limit results to 20 samples
  """

  results = sqldb_agent.invoke({"question": question})
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
  