from langchain_core.prompts import (
  ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
)

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from typing import Dict, Literal

from agent.chart.utils import run_python_fig_code
from agent.chart.prompts import *
from agent.models import gpt4

LLM = gpt4
NTRY = 10

def initialize(state: Dict, config: Dict) -> Dict:
  """ Checks for valid state input and initializes state variables"""
  verbose = config.get("verbose", False)
  
  if verbose:
    print("--- Initializing ---")
  
  # Check for existence of dataset 
  if config.get("df") is None:
    raise Exception("Must provide dataframe `df` in config")
  
  if state.get("messages") is None:
    state["messages"] = []
  
  # reset debug attempts  
  state["ntry"] = 0
  
  # reset result
  state["results"] = {}
  
  return state


def route(state: Dict, config: Dict) -> Literal["debug", "summarize"]:
  """ Routes agent action to either debugger or program end """
  
  if state.get("results").get("fig") is None:
    return "summarize" if state["ntry"] > config.get("ntry", NTRY) else "debug"
  else:
    return "summarize"


def generate_code(state: Dict, config: Dict) -> Dict:
  """ Provides natural language description of chart generated by agent. """
  if config.get("verbose", False):
    print("--- Generating figure code ---")
    
  # call llm to generate figure code 
  llm = config.get("llm", LLM)
  prompt = PromptTemplate.from_template(MAKE_CHART_CODE_PROMPT)
  chain = prompt | llm | StrOutputParser()
  
  state["column_names"] = config.get("df")[0].columns 
  result = chain.invoke(state)
  
  # update state
  state["results"]["code"] = result.replace("fig.show()", "")
  return state


def run_code(state: Dict, config: Dict) -> Dict:
  """ Runs figure generating code block and returns result or error. """  
  if config.get("df")[0] is None:
    raise Exception("No active data")

  state["ntry"] += 1
  
  if config.get("verbose", False):
    print("--- Running figure code ---")
    print(f"Attempt {state['ntry']}")
  
  code = state["results"]["code"]
  result = run_python_fig_code(code, {"df": config.get("df")[0]}) 
  
  if isinstance(result, Exception):
    result = f"Failed to execute CODE:\n\n{code}.\n\nERROR: {repr(result)}"
    state["messages"] = [AIMessage(result)]
  else:
    state["messages"] = [AIMessage("SUCCESS!")]
    state["results"]["fig"] = result

  return state


def debug_code(state: Dict, config: Dict) -> Dict:
  """ Provides natural language description of chart generated by agent. """
  
  if config.get("verbose", False):
    print("--- Debugging figure code ---")
    
  # call llm to generate figure code 
  llm = config.get("llm", LLM)
  
  prompt = ChatPromptTemplate.from_messages([
    ("system", DEBUG_CHART_CODE_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
  ])   
  
  chain = prompt | llm | StrOutputParser()

  # call chain and update code state 
  _state = {
    "column_names": config.get("df")[0].columns, 
    "code": state["results"]["code"],
    "messages":state["messages"]
  }
  
  state["results"]["code"] = chain.invoke(_state).replace("fig.show()", "")
  return state 


def summarize_chart(state: Dict, config: Dict) -> Dict:
  """ Provides natural language description of chart generated by agent. """
  
  if config.get("verbose", False):
    print("--- Summarizing figure code ---")

  llm = config.get("llm", LLM)    
  prompt = PromptTemplate.from_template(SUMMARIZE_CHART_PROMT)  
  chain = prompt | llm | StrOutputParser()  
  
  code = state["results"]["code"]
  final_message = state["messages"][-1].content
  _state = {"code":code, "message": final_message}
  state["results"]["summary"] = chain.invoke(_state)
  return state