from agent.base_agents import chat_agent
from agent.models import gpt4
from agent.geordi.prompts import *

# Initialize SQLDB base agent
from agent.sqldb.agent import SQLDBAgent
from agent.chart.agent import ChartAgent
from agent.geordi.utils import *

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser
from typing import Literal, Dict

LLM = gpt4

def initialize(state: Dict, config: Dict) -> Dict:
  """ Initialize graph state values """
  
  # Initialize state values
  check_config(config)
  
  if state.get("sqldb_agent") is None:
    state["sqldb_agent"] = SQLDBAgent() 

  if state.get("chart_agent") is None:
    state["chart_agent"] = ChartAgent()
    
  if state.get("messages") is None:
    state["messages"] = []
    
  if state.get("ntry") is None:
    state["ntry"] = 0
    
  return state


def load_data(state: Dict, config: Dict) -> Dict:
  """ Calls sqldb agent to load data """
  check_config(config)
  sqldb_agent = state["sqldb_agent"] 

  if config.get("verbose", False):
    print("--- LOADING DATASET ---")

  # Run sql agent to load data
  invoke_state = {"question": state.get("question")}
  sqldb_agent.state = sqldb_agent.invoke(invoke_state, config)   
  state["sqldb_agent"] = sqldb_agent
  state["ntry"] += 1

  # Add messages from sql agent to chat thread
  if sqldb_agent.get("df") is not None:
    summary = sqldb_agent.get("df_summary")
    msg = f"Successfully loaded the following table: {summary}"
    state["messages"] = [AIMessage(msg, name=sqldb_agent.name)]
  else:
    msg = f"Failed to load table."
    state["messages"] = [AIMessage(msg, name=sqldb_agent.name)]
    
  return state


def route(state: Dict, config: Dict) -> Literal["figure", "analysis"]:
  """ Determine which downstream task to be performed"""
  # TODO: add handling for "Failed to load data setting"
  llm = config.get("llm", LLM)
  sqldb_agent = state["sqldb_agent"] 
  question = state.get("question")
  relevance = check_relevance(question, sqldb_agent, llm)
  
  if config.get("verbose", False):
    print("--- ROUTING ---")
    print(relevance)

  # If data cannot address request, send back to explainer
  if relevance == "insufficient":
    return "end"
  
  chain = chat_agent(llm, TASK_PROMPT)
  invoke_state = {"question": question, "messages":state["messages"]}
  return chain.invoke(invoke_state).content 


def explain_router(state: Dict, config: Dict) -> Dict:
  """ Provide a natural language description of why data are insufficient"""
  llm = config.get("llm", LLM)
  sqldb_agent = state["sqldb_agent"] 
  
  chain = chat_agent(llm, EXPLAIN_ROUTER_PROMPT)
  
  invoke_state = {
    "question": state["question"],
    "messages": state["messages"],
    "table": sqldb_agent.get("df").head().to_string(),
    "df_summary": sqldb_agent.get("df_summary"),
    "query": sqldb_agent.get("query")
  }

  result = chain.invoke(invoke_state)
  state["messages"] = [AIMessage(result.content)]
  return state 
  

def make_figure(state: Dict, config: Dict) -> Dict:
  """ Calls chart agent to generate a figure """
  chart_agent = state["chart_agent"]  
  sqldb_agent = state["sqldb_agent"] 

  if config.get("verbose", False):
    print("--- GENERATING FIGURE ---")
  
  config["df"] = [sqldb_agent.get("df")]
  invoke_state = {"question": state["question"]}
  chart_agent.state = chart_agent.invoke(invoke_state, config=config)
  state["chart_agent"] = chart_agent
  return state

def run_analysis(state: Dict)  -> Dict:
  """ Calls analysis agent to run analysis"""
  message = "Analysis engine not curently supported"
  state["messages"] = [AIMessage(message)]
  return state
