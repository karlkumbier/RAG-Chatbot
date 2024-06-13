from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, StateGraph

from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph

from langgraph.prebuilt import ToolNode
from agent.models import cold_gpt35

import functools
from langchain_core.messages import AIMessage
import operator
from typing import Annotated, Sequence, TypedDict, Literal
from typing_extensions import TypedDict
from agent.sqldb.utils import sql_tool_agent

llm = cold_gpt35
username = "kkumbier"
password = "persisters"
port = "5432"
host = "localhost"
db = "persisters"

pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db}"
db = SQLDatabase.from_uri(pg_uri)
sql_toolkit = SQLDatabaseToolkit(db=db, llm=cold_gpt35)


# Helper function to create a node for a given agent
def agent_node(state, agent, name):
  """ Wrapper function for creating node from tool calling agent """
  result = agent.invoke(state)
  
  # Convert agent output into a format suitable to append to the global state
  if isinstance(result, ToolMessage):
    pass
  else:
    result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
  
  state = {"messages": [result], "sender": name}
  return state

# Initialize router node
def router(state) -> Literal["call_tool", "__end__"]:
    """ Routes agent action to either tool call or program end """
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


# Initialize database node
#system_message = "Query the database to answer user questions"
db_agent = sql_tool_agent(llm, sql_toolkit)
db_node = functools.partial(agent_node, agent=db_agent, name="database")

# Initialize tool node
tool_node = ToolNode(sql_toolkit.get_tools())

# Initialize graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str

workflow = StateGraph(AgentState)
workflow.add_node("database", db_node)
workflow.add_node("call_tool", tool_node)

workflow.set_entry_point("database")

workflow.add_conditional_edges(
    "database",
    router,
    {"call_tool": "call_tool", "__end__": END},
)

workflow.add_edge("call_tool", "database")

#workflow.add_edge("call_tool", "database")
#workflow.set_entry_point("database")
graph = workflow.compile()
graph.get_graph().print_ascii()

question = "Return the table of log2 fold change, p-value, and SYMBOL from the 122023001-RNASEQ-CELL screen. Show the results for 10 samples"
events = graph.invoke({"messages": [HumanMessage(question)]})

am = [m for m in events["messages"] if type(m) is AIMessage]
tm = [m for m in events["messages"] if type(m) is ToolMessage]

result = db_agent.invoke({"messages":events["messages"]})
result