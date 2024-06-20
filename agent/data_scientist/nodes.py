from agent.base_agents import chat_agent
from agent.models import gpt4
from agent.data_scientist.prompts import *
from agent.data_scientist.utils import *
from agent.sqldb.agent import SQLDBAgent
from agent.chart.agent import ChartAgent

from langchain_core.messages import AIMessage
from typing import Literal, Dict

LLM = gpt4
SQLDB_AGENT = SQLDBAgent()
CHART_AGENT = ChartAgent()

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
    
  state["results"] = {}
  return state


def load_data(state: Dict, config: Dict) -> Dict:
  """ Calls sqldb agent to load data """
  check_config(config)

  state["ntry"] += 1
  if config.get("verbose", False):
    print("--- LOADING DATASET ---")
    print(f"Attempts {state['ntry']}")

  # Run sql agent to load data
  invoke_state = {"question": state.get("question")}
  sqldb_state = SQLDB_AGENT.invoke(invoke_state, config)   

  # Add messages from sql agent to chat thread
  if sqldb_state.get("results").get("df") is not None:
    summary = sqldb_state.get("results").get("summary")
    msg = f"Successfully loaded the following table:\n\n{summary}"
    state["messages"] = [AIMessage(msg, name=SQLDB_AGENT.name)]
  else:
    msg = f"Failed to load table."
    state["messages"] = [AIMessage(msg, name=SQLDB_AGENT.name)]
    
  state["sqldb_state"] = sqldb_state
  return state


def route(state: Dict, config: Dict) -> Literal["figure", "analysis"]:
  """ Determine which downstream task to be performed"""
  # TODO: add handling for "Failed to load data setting"
  llm = config.get("llm", LLM)
  relevance = check_relevance(state["question"], state["sqldb_state"], llm)
  
  if config.get("verbose", False):
    print("--- ROUTING ---")
    print(relevance)

  # If data cannot address request, send back to explainer
  if relevance == "insufficient":
    return "end"
  
  prompt = PromptTemplate.from_template(TASK_PROMPT)
  chain = prompt | llm | StrOutputParser()
  return chain.invoke(state)


def explain_router(state: Dict, config: Dict) -> Dict:
  """ Provide a natural language description of why data are insufficient"""
  llm = config.get("llm", LLM)
  sqldb_state = state["sqldb_state"] 
  
  chain = chat_agent(llm, EXPLAIN_ROUTER_PROMPT)
  
  _state = {
    "question": state["question"],
    "messages": state["messages"],
    "table": sqldb_state.get("results").get("df").head().to_string(),
    "df_summary": sqldb_state.get("results").get("summary"),
    "query": sqldb_state.get("results")("query")
  }

  result = chain.invoke(_state)
  state["messages"] = [AIMessage(result.content, name="explanation")]
  return state 
  

def make_figure(state: Dict, config: Dict) -> Dict:
  """ Calls chart agent to generate a figure """

  if config.get("verbose", False):
    print("--- GENERATING FIGURE ---")
  
  df = state["sqldb_state"]["results"]["df"] 
  if df is None:
    raise Exception("Data table not laoded")
  
  config["df"] = [df]
  _state = {"question": state["question"]}
  state["chart_state"] = CHART_AGENT.invoke(_state, config=config)
  state["results"] = state["chart_state"]["results"]
  return state


def make_table(state: Dict, config: Dict) -> Dict:
  """ Sets results to loaded table"""
  state["results"] = state["sqldb_state"]["results"]
  return state


def run_analysis(state: Dict, config: Dict) -> Dict:
  """ Calls analysis agent to run analysis"""
  message = "Analysis engine not curently supported"
  state["messages"] = [AIMessage(message)]
  state["results"] = None
  return state
