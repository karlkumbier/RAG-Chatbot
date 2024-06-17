from agent.sqldb.sqldb_agent import sqldb_agent
from agent.chart.chart_agent import chart_agent
from agent.base_agents import chat_agent, agent_node
from agent.models import cold_gpt35, gpt4
from agent.geordi.prompts import *
from langgraph.graph import StateGraph

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from typing import Sequence, Annotated, Literal, Dict
from typing_extensions import TypedDict
from plotly.graph_objects import Figure
import operator
import pandas as pd
import functools

LLM = gpt4

question = """Generate a figure of differential expression p-value versus log2 
fold change for PC9 cell lines. The differential expression analysis should
contrast SOC with no drug. Use data from the hypoxia vs. normoxia screen. 
"""


# TODO: Agent class - initialized with "name" that appends key to AgentState
# TODO: data_loader should incorporate message from insufficiency explainer
# TODO: Fig explanation of generated figures
# TODO clean up

##############################################################################
# Agent assesses relevance of available data for downstream analysis. If data 
# sufficient, send on for processing.
##############################################################################
# Define agent state
class AgentState(TypedDict):
  question: str # user question
  messages: Annotated[Sequence[BaseMessage], operator.add] # chat history
  db_query: str # last SQL query
  fig: Figure # generated figure
  df: pd.DataFrame # dataframe from query 
  df_summary: str # description of dataframe
  ntry: int # number of debug attemps


# Define graph node functions
def initializer_node(state: Dict) -> Dict:
  """ Initialize tools and evaluate DB schema"""
  if state.get("messages") is None:
    state["messages"] = []
    
  if state.get("ntry") is None:
    state["ntry"] = 0
    
  return state

def load_data(state: Dict, config: Dict) -> Dict:
  """ Calls sqldb agent to load data """
  
  if config.get("verbose", False):
    print("--- LOADING DATASET ---")
    
  # TODO: maybe need to modify state based to DB agent
  result = sqldb_agent.invoke(state)
  state["df_summary"] = result["df_summary"]
  state["df"] = result["df"] 
  state["db_query"] = result["db_query"]
  state["ntry"] += 1
  
  if config.get("verbose", False):
    print(state["df_summary"])
    print(state["db_query"])
    
  return state


def check_relevance(state: Dict) -> Literal["sufficient", "insufficient"]:
  """ Determines whether data are sufficient or insufficient for request"""
  agent = chat_agent(LLM, RELEVANCE_PROMPT)
  return agent.invoke(state).content

def router(state: Dict, config: Dict) -> Literal["figure", "analysis"]:
  """ Determine which downstream task to be performed"""
  
  relevance = check_relevance(state)
  
  if config.get("verbose", False):
    print("--- ROUTING ---")
    print(relevance)

  # If data cannot address request, send back to laoder
  if relevance == "insufficient":
    return "insufficiency_explainer"
  
  task_agent = chat_agent(LLM, TASK_PROMPT)
  return task_agent.invoke(state).content 

def explain_insufficient(state: Dict) -> Dict:
  
  agent = chat_agent(LLM, EXPLAIN_IRRELEVANCE_PROMPT)
  state["table"] = state["df"].head().to_string()
  result = StrOutputParser(agent.invoke(state))
  state["messages"] = [result]
  return state 
  
def make_figure(state: Dict, config: Dict) -> Dict:
  """ Calls chart agent to generate a figure """
  
  if config.get("verbose", False):
    print("--- GENERATING FIGURE ---")
  
  result = chart_agent.invoke(state)
  state["fig"] = result["fig"]
  return state

def run_analysis(state: Dict)  -> Dict:
  """ Calls analysis agent to run analysis"""
  message = "Analysis engine not curently supported"
  result = AIMessage(message)
  state["messages"] = [result]
  return state

# Construct agent graph
workflow = StateGraph(AgentState)
workflow.add_node("initializer", initializer_node)
workflow.add_node("data_loader", load_data)
workflow.add_node("insufficiency_explainer", explain_insufficient)
workflow.add_node("figure_generator", make_figure)
workflow.add_node("analysis_runner", run_analysis)

workflow.set_entry_point("initializer")
workflow.add_edge("initializer", "data_loader")
workflow.add_edge("insufficiency_explainer", "data_loader")
workflow.add_edge("figure_generator", "__end__")
workflow.add_edge("analysis_runner", "__end__")

workflow.add_conditional_edges(
  "data_loader",
  router,
  {
    "insufficiency_explainer": "insufficiency_explainer", 
    "figure": "figure_generator", 
    "analysis": "analysis_runner"
  }
)

geordi = workflow.compile()
geordi.get_graph().print_ascii()

config = {"verbose": True}
result = geordi.invoke({"question": question}, config=config)

result["fig"].show()


##############################################################################
# If data are not relevant, describe problem, retry query
##############################################################################

##############################################################################
# Once data are relevant...
##############################################################################

##############################################################################
# Determine task to be performed - task should be in state, to be read by 
# supervising agents
##############################################################################

##############################################################################
# Carry out task
# 1. Update figure agent according to state / table summary / new changes
# 2. Add node for `analysis` to return message: methods not implemented yet
# 3. Modify state (e.g., by adding fig). Add state message summarizing the fig.
##############################################################################

##############################################################################
# Assess task completeness - chat history / single message node
##############################################################################

##############################################################################
# If task not complete, describe problem, route description to one of database # loader or figure generator and rerun cycle
##############################################################################

