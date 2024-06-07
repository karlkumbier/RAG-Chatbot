from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    AIMessage
)

from typing_extensions import TypedDict
from typing import Sequence, Annotated, Literal
from agent.models import cold_gpt35
from agent.sqldb.utils import *
from agent.sqldb.prompts import SQL_TOOL_PROMPT
import functools
import operator

def agent_node(state, agent, name):
  """ Wrapper function for creating node from tool calling agent """
  print(f"--- Running: {name} ---")
  result = agent.invoke(state)
  
  # Convert agent output into a format suitable to append to the global state
  if isinstance(result, ToolMessage):
    pass
  else:
    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
  print(result)
  state["messages"] = [result]
  state["sender"] =  name
  return state

def router(state) -> Literal["call_tool", "__end__"]:
    """ Routes agent action to either tool call or program end """
    print("--- Routing ---")
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"
    if "SUCCESS" in last_message.content:
        # Agent decided the work is done
        return "__end__"
    else:
      Warning("Agent failed to execute query")
      return "__end__"

# Initialize agent nodes for sql query generation and tool calling
username = "kkumbier"
password = "persisters"
port = "5432"
host = "localhost"
db = "persisters"

# Initialize connection to database
pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db}"
db = SQLDatabase.from_uri(pg_uri)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=cold_gpt35)

# Node for tool calling agent - generates tool inputs
db_node = functools.partial(
  agent_node, 
  agent=tool_agent(cold_gpt35, SQL_TOOL_PROMPT, sql_toolkit), 
  name="database"
)

# Node for tool calling - uses tool with input from agent
tool_node = ToolNode(sql_toolkit.get_tools())

###############################################################################
# Initialize agent graph 
###############################################################################
# TODO: create user that does not have write permissions
# TODO: rewrite from csv file, may have been overwritten in testing
# TODO: pull in comments for schema generation
#   SELECT oid, relname, description FROM pg_class
#   INNER JOIN pg_description ON pg_class.oid = pg_description.objoid;

# Set graph edges
class AgentState(TypedDict):
  question: str
  messages: Annotated[Sequence[BaseMessage], operator.add]
  sender: str

workflow = StateGraph(AgentState)
workflow.add_node("database", db_node)
workflow.add_node("call_tool", tool_node)

workflow.set_entry_point("database")
workflow.add_edge("call_tool", "database")

workflow.add_conditional_edges(
    "database",
    router,
    {"call_tool": "call_tool", "__end__": "__end__"},
)

agent = workflow.compile()
agent.get_graph().print_ascii()


if __name__ == "__main__":

  question = """
    Get the table of log2 fold change, p-value, and SYMBOL from the 
    122023001-RNASEQ-CELL screen. Limit results to 10 samples
  """
  
  results = agent.invoke({
    "question": question,
    "messages": []
  })

  queries = [m for m in results["messages"] if isinstance(m, AIMessage)]
  final_query = queries[-2]
  qry = final_query.additional_kwargs["tool_calls"][0]["function"]["arguments"]
  
  print(f"SQL QUERY: {qry}")
  print(results["messages"][-1].content)