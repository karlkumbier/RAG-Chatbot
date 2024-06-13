from langchain.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage

from typing_extensions import TypedDict
from typing import Sequence, Annotated, Literal, Dict
from agent.models import cold_gpt35
from agent.base_agents import chat_agent, agent_node
from agent.sqldb.utils import set_schema_node_, run_query_node_
from agent.sqldb.prompts import SQL_QUERY_PROMPT, SQL_DEBUGGER_PROMPT
import functools
import operator
import pandas as pd

# Initialize base parameters for agent
LLM = cold_gpt35
NTRY = 10

# Initialize connection to database 
# TODO: ** add code parser to extract sql query from response...
# TODO: create user that does not have write permissions
# TODO: rewrite from csv file, may have been overwritten in testing
# TODO: pull in comments for schema generation
#   SELECT oid, relname, description FROM pg_class
#   INNER JOIN pg_description ON pg_class.oid = pg_description.objoid;
username = "kkumbier"
password = "persisters"
port = "5432"
host = "localhost"
db = "persisters"

pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db}"
db = SQLDatabase.from_uri(pg_uri)

schema_query = """
  SELECT oid, relname, description FROM pg_class
  INNER JOIN pg_description ON pg_class.oid = pg_description.objoid;
"""

import pandas as pd
result = result = pd.read_sql(schema_query, db._engine)

# Set agent initialer and router
def initializer_node(state: Dict) -> Dict:
  """ Initialize tools and evaluate DB schema"""
  if state.get("messages") is None:
    state["messages"] = []
    
  if state.get("ntry") is None:
    state["ntry"] = 0
    
  return state
  
def router(state) -> Literal["debug", "__end__"]:
  """ Routes agent action to either debug or program end """ 
  if state.get("df") is None:
    return "__end__" if state["ntry"] > NTRY else "debug"
  else:
    return "__end__"

# Node for tool calling agent - generates tool inputs
sql_query_node = functools.partial(
  agent_node, 
  agent=chat_agent(cold_gpt35, SQL_QUERY_PROMPT), 
  name="sql_query_generator"
)

sql_query_debug_node = functools.partial(
  agent_node,
  agent=chat_agent(LLM, SQL_DEBUGGER_PROMPT),
  name="sql_debugger"
)

run_query_node = functools.partial(
  run_query_node_,
  db=db,
  name="sql_executor"  
)

set_schema_node = functools.partial(
  set_schema_node_,
  llm=LLM,
  db=db
)

###############################################################################
# Initialize agent graph 
###############################################################################
class AgentState(TypedDict):
  question: str
  messages: Annotated[Sequence[BaseMessage], operator.add]
  db_query: str
  df: pd.DataFrame
  db_schema: str
  dialect: str
  sender: str
  ntry: int

workflow = StateGraph(AgentState)
workflow.add_node("initializer", initializer_node)
workflow.add_node("set_schema", set_schema_node) # make this
workflow.add_node("sql_query_generate", sql_query_node) # chat agent
workflow.add_node("run_query", run_query_node) # make this
workflow.add_node("sql_query_debug", sql_query_debug_node) # chat agent

workflow.set_entry_point("initializer")
workflow.add_edge("initializer", "set_schema")
workflow.add_edge("set_schema", "sql_query_generate")
workflow.add_edge("sql_query_generate", "run_query")
workflow.add_edge("sql_query_debug", "run_query")

workflow.add_conditional_edges(
  "run_query",
  router,
  {"debug": "sql_query_debug", "__end__": "__end__"},
)

db_agent = workflow.compile()
#db_agent.get_graph().print_ascii()


if __name__ == "__main__":

  question = """
    Get a tables of log2 fold change, p-value, and SYMBOL from the 122023001 and 012023001 screens. Filter these to PC9 cell lines and join the two tables by symbol. Limit results to 20 samples
  """

  results = db_agent.invoke({"question": question})
  print(results["db_query"])
  results["df"]
  query = results["db_query"]
  output = db.run(query)