from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage

from typing_extensions import TypedDict
from typing import Sequence, Literal, Annotated

from agent.models import gpt4
from agent.chart.prompts import CHART_PROMPT, DEBUG_CHART_PROMPT
from agent.base_agents import chat_agent, agent_node
from agent.chart.utils import make_chart_node_

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


def router(state) -> Literal["debug", "__end__"]:
  """ Routes agent action to either debugger or program end """
  
  if state.get("fig") is None:
    return "__end__" if state["ntry"] > NTRY else "debug"
  else:
    return "__end__"


# Initialize agents nodes for chart making & debugging
chart_code_node = functools.partial(
  agent_node, 
  agent=chat_agent(gpt4, CHART_PROMPT), 
  name="chart_code"
)

debug_code_node = functools.partial(
  agent_node, 
  agent=chat_agent(gpt4, DEBUG_CHART_PROMPT), 
  name="debug_code"
)

make_chart_node = functools.partial(
  make_chart_node_,
  name="make_chart"
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
workflow.add_node("chart_code", chart_code_node)
workflow.add_node("make_chart", make_chart_node)
workflow.add_node("debug_code", debug_code_node)

workflow.set_entry_point("initializer")
workflow.add_edge("initializer", "chart_code")
workflow.add_edge("chart_code", "make_chart")
workflow.add_edge("debug_code", "make_chart")

workflow.add_conditional_edges(
    "make_chart",
    router,
    {"debug": "debug_code", "__end__": "__end__"},
)

chart_agent = workflow.compile()
#chart_agent.get_graph().print_ascii()

if __name__ == "__main__":
  base_dir = "/awlab/projects/2021_07_Persisters/data/"
  data_dir = "012023001-RNASEQ-CELL/level2"
  df = pd.read_csv(os.path.join(base_dir, data_dir, "gene_de.csv"))
  
  question =  """
  Generate a plot log2 fold change on the x-axis and p-value on the y-axis. Filter data to include only PC9 cell line and the SOC, Normoxia v. No drug, Normoxia ContrastFull.
  """ 
  
  result = chart_agent.invoke({"question": question, "df": df})
  messages = result["messages"]
  print(messages[-1].content)
  result["fig"].show()