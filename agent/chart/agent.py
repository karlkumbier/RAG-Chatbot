from langgraph.graph import StateGraph
from langchain_core.messages import BaseMessage

from typing_extensions import TypedDict
from typing import Sequence, Annotated, Dict

from agent.chart import nodes

from plotly.graph_objects import Figure
import operator

class ChartAgentState(TypedDict):
    question: str # user question
    messages: Annotated[Sequence[BaseMessage], operator.add] # chat history
    code: str # figure code
    fig: Figure # figure object
    fig_summary: str # plain text description of figure
    ntry: int # number of attempts

class ChartAgent:
  
  def __init__(self, name="chart"):
    self.agent = self.__build_graph__()
    self.name = name
    self.state = None
    
  def __build_graph__(self):
    workflow = StateGraph(ChartAgentState)
    workflow.add_node("initializer", nodes.initialize)
    workflow.add_node("code_generator", nodes.generate_code)
    workflow.add_node("code_runner", nodes.run_code)
    workflow.add_node("code_debugger", nodes.debug_code)
    workflow.add_node("chart_summarizer", nodes.summarize_chart)

    workflow.set_entry_point("initializer")
    workflow.add_edge("initializer", "code_generator")
    workflow.add_edge("code_generator", "code_runner")
    workflow.add_edge("code_debugger", "code_runner")
    workflow.add_edge("chart_summarizer", "__end__")
    
    workflow.add_conditional_edges(
        "code_runner",
        nodes.route,
        {"debug": "code_debugger", "summarize": "chart_summarizer"},
    )

    return workflow.compile()
    
  def __print__(self):
    if self.agent is None:
      raise Exception("Agent graph not built")
    
    self.agent.get_graph().print_ascii()
    
  def invoke(self, state: Dict, config: Dict):
    return self.agent.invoke(state, config)
  
  def __clear_state__(self):
    self.state = None
    
  def get(self, key: str):
    if self.state is None:
      return None
    else:
      return self.state.get(key)


if __name__ == "__main__":
  from agent.chart.agent import *
  from agent.models import gpt4
  import pandas as pd 
  import os 
  
  base_dir = "/awlab/projects/2021_07_Persisters/data/"
  data_dir = "012023001-RNASEQ-CELL/level2"
  df = pd.read_csv(os.path.join(base_dir, data_dir, "gene_de.csv"))
  
  # Initialize chart agent
  chart_agent = ChartAgent()
  
  # Test 1
  question =  """
  Generate a plot log2 fold change on the x-axis and p-value on the y-axis. 
  Filter data to include only PC9 cell line and the SOC, Normoxia v. No drug, 
  Normoxia ContrastFull.
  """ 
  
  config = {"name": chart_agent.name, "llm": gpt4, "df": [df], "verbose": True}
  result = chart_agent.invoke({"question": question}, config)

  name = chart_agent.name
  code = result[f"{name}_code"]
  print(result[f"{name}_code"])
  print(result[f"{name}_fig_summary"]) 