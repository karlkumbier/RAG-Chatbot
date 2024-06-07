from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage, AIMessage

from typing_extensions import TypedDict
from typing import Sequence, Literal, Annotated

from agent.models import gpt4
from agent.chart.utils import *
from agent.chart.prompt import CHART_PROMPT, DEBUG_CHART_PROMPT
from plotly.graph_objects import Figure

import pandas as pd
import os
import functools
import operator

NTRY = 5

def initializer_node(state):
  """ Checks for valid state input and initializes state variables"""
  
  # Check for existence of dataset
  if state.get("df") is None:
    raise Exception("No active data")
  else:
    state["column_names"] = state["df"].columns
    
  # Check for message history and set to empty if missing
  if state.get("messages") is None:
    state["messages"] = []
    
  state["ntry"] = 0
  return state

def agent_node(state, agent, name):
  """ Wrapper function for message handling agent """
  if state["df"] is None:
    raise Exception("No active data")
  else:
    state["column_names"] = state["df"].columns
  
  result = agent.invoke(state)
  result = AIMessage(**result.dict(exclude={"type", "name"}), name=name)
  ntry = state["ntry"] + 1
  
  # Convert agent output into a format suitable to append to the global state
  state["messages"] = [result]
  state["sender"] = name
  state["ntry"] = ntry
  
  return state


def code_executor_node(state):
  """ Runs and returns results from code block """
  code = state["messages"][-1].content
  
  if state["df"] is None:
    raise Exception("No active data")
  else:
    _globals = {"df": state["df"]}
    result = execute_python_code(code, _globals) 
  
  
  if isinstance(result, Exception):
    result = f"Failed to execute CODE:\n\n{code}.\n\nERROR: {repr(result)}"
    state["messages"] = [AIMessage(result, name="code_Executor")]
  else:
    state["code"] = code
    state["fig"] = result
  
  return state


def router(state) -> Literal["debug", "__end__"]:
  """ Routes agent action to either debugger or program end """
  
  if state.get("fig") is None:
    return "__end__" if state["ntry"] > NTRY else "debug"
  else:
    return "__end__"


# Initialize agents nodes for chart making & debugging
chart_node = functools.partial(
  agent_node, 
  agent=chat_agent(gpt4, CHART_PROMPT), 
  name="chart_maker"
)

debug_node = functools.partial(
  agent_node, 
  agent=chat_agent(gpt4, DEBUG_CHART_PROMPT), 
  name="debugger"
)

###############################################################################
# Initialize agent graph
###############################################################################
# Set graph edges
class AgentState(TypedDict):
  question: str
  messages: Annotated[Sequence[BaseMessage], operator.add]
  sender: str
  ntry: int
  column_names: str
  df: pd.DataFrame
  fig: Figure

workflow = StateGraph(AgentState)
workflow.add_node("initializer", initializer_node)
workflow.add_node("chart_maker", chart_node)
workflow.add_node("code_executor", code_executor_node)
workflow.add_node("debugger", debug_node)

workflow.set_entry_point("initializer")
workflow.add_edge("initializer", "chart_maker")
workflow.add_edge("chart_maker", "code_executor")
workflow.add_edge("debugger", "code_executor")

workflow.add_conditional_edges(
    "code_executor",
    router,
    {"debug": "debugger", "__end__": "__end__"},
)

agent = workflow.compile()
agent.get_graph().print_ascii()

if __name__ == "__main__":
  base_dir = "/awlab/projects/2021_07_Persisters/data/"
  data_dir = "012023001-RNASEQ-CELL/level2"
  df = pd.read_csv(os.path.join(base_dir, data_dir, "gene_de.csv"))
  question =  "plot log2 fold change on the x-axis and p-value on the y-axis" 
  
  result = agent.invoke({"question": question, "df": df})
  messages = result["messages"]
  
  #out = chart_agent.invoke({"question":question, "column_names":df.columns})
