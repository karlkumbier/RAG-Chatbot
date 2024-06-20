from typing import Dict, Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from langchain_core.prompts import (
  ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
)

from agent.models import cold_gpt35
from agent.sqldb.utils import *
from agent.sqldb.prompts import *

import pandas as pd
LLM = cold_gpt35
NTRY = 10

def initialize(state: Dict, config: Dict) -> Dict:
  """ Initialize baseline state variables """
  # Check connection to database
  db = config.get("db")
  
  if db is None:
    raise Exception("No database specified in config")
  
  if state.get("messages") is None:
    state["messages"] = []
    
  state["ntry"] = 0
  state["results"] = {}
  state["dialect"] = db.dialect
  
  llm = config.get("llm", LLM)
  state["dbschema"] = build_schema(state["question"], db, llm) 
  return state


def generate_query(state: Dict, config: Dict) -> Dict:
  """ Generates database query to answer user request"""
  llm = config.get("llm", LLM)
  verbose = config.get("verbose", False)
  
  if verbose:
    print(" --- Generating SQL query ---")
    
  # call llm to generate figure code
  prompt = ChatPromptTemplate.from_messages([
    ("system", GENERATE_QUERY_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
  ])   
  
  chain = prompt | llm | StrOutputParser()
  invoke_state = {k.replace("", ""):v for k, v in state.items()}
  result = chain.invoke(invoke_state)
  
  # update state
  state["results"]["query"] = result
  return state


def run_query(state: Dict, config: Dict) -> Dict:
  """ Runs SQL query and returns result as pd.DataFrame """
  db = config.get("db")
  verbose = config.get("verbose", False) 
  state["ntry"] += 1
  
  if verbose:
    print(" --- Checking query ---")
    print(f"Attempts: {state['ntry']}")
    
  # Check connection to database
  if db is None:
    raise Exception("No database spcified in config")
  
  # Extract sql query from code block
  block_query = state["results"]["query"] 
  query = extract_sql_query(block_query)
  
  # Try to run extracted code, raise exceptions if unsuccessful
  if query is None:
    result = SyntaxError(
      f"""
        Failed to execute QUERY:\n\n{block_query}
        
        Perhaps you forgot to wrap the QUERY in ```. 
        Example format: ```query```
      """
    )
  else:
    try:
      result = pd.read_sql(query, db._engine)
    except BaseException as e:
      result = e
    
  if isinstance(result, Exception):
    result = f"Failed to execute QUERY:\n\n{query}.\n\nERROR: {repr(result)}"
    state["messages"] = [AIMessage(result)]
  elif len(result.index) == 0:
    result = f"QUERY:\n\n{query} returned no results.\n\nERROR: empty return"
  else:
    state["messages"] = [AIMessage("SUCCESS!")]
    state["results"]["df"] = result
    state["results"]["query"] = query
    
  return state


def route(state: Dict, config: Dict) -> Literal["debug", "__end__"]:
  """ Routes agent action to either debug or table summary """ 
  if state.get("results").get("df") is None:
    return "summarize" if state["ntry"] > config.get("ntry", NTRY) else "debug"
  else:
    return "summarize"


def debug_query(state: Dict, config: Dict) -> Dict:
  """ Generates database query to answer user request"""
  llm = config.get("llm", LLM)
  verbose = config.get("verbose", False)
  
  if verbose:
    print("--- Debugging query ---")
    
  # call llm to generate figure code
  prompt = ChatPromptTemplate.from_messages([
    ("system", DEBUG_QUERY_PROMPT),
    MessagesPlaceholder(variable_name="messages"),
  ])   
  
  chain = prompt | llm | StrOutputParser()
  state["query"] = state["results"]["query"]
  result = chain.invoke(state)
  
  # update state
  state["results"]["query"] = result
  return state


def summarize_table(state: Dict, config: Dict) -> Dict:
  """ Generates plain text description of table generated from SQL query""" 
  llm = config.get("llm", LLM)
  verbose = config.get("verbose", False)  
  
  if verbose:
    print("--- Summarizing query ---")
    
  if state.get("results").get("query") is None:
    state["results"]["summary"] = "Did not generate query successfully"
  else:
    prompt = PromptTemplate.from_template(template=SUMMARIZE_TABLE_PROMPT)
    chain = prompt | llm | StrOutputParser()

    # Clean state names for prompt invoking
    state["table_string"] = state["results"]["df"].head().to_string()
    state["query"] = state["results"]["query"]  
    state["results"]["summary"] = chain.invoke(state)
  
  return state  

