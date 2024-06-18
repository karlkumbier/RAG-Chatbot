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
from agent.base_agents import agent_node, chat_agent

import functools
import pandas as pd
import json
LLM = cold_gpt35
NTRY = 10

def initialize(state: Dict, config: Dict) -> Dict:
  """ Initialize baseline state variables """
  name = config.get("name", "sqldb")
  db = config.get("db")

  # Check connection to database
  if db is None:
    raise Exception("No database specified in config")
  
  if state.get(f"{name}_messages") is None:
    state[f"{name}_messages"] = []
    
  if state.get(f"{name}_ntry") is None:
    state[f"{name}_ntry"] = 0
    
  return state


def set_schema(state: Dict, config: Dict) -> Dict:
  """ Queries database and initializes graph state schema. """
  
  db = config.get("db")
  name = config.get("name", "sqldb")
  llm = config.get("llm", LLM)
  verbose = config.get("verbose", False)
  
  if verbose:
    print("--- Initialization DB schema ---")
    
  # Check connection to database
  if db is None:
    raise Exception("No database specified in config")

  # Query SQL database for description table: comments on tables/cols
  tables = make_table_list(db) 
  table_summary = {}
  
  for table in tables:  
    table_comments = get_table_comments(db, table)
    column_comments = get_column_comments(db, table)
    
    table_summary[table] = make_table_summary(
      table, table_comments, column_comments, db
    ) 
  
  # Select tables that are relevant to the query
  tables_select = select_table(llm, state[f"question"], table_summary)
  tables_select = tables_select.split(", ")
  
  if tables_select == "None":
    raise Exception("Tables relevant to the question cannot be found")

  # Initialize schema for relevant tables
  table_summary = {k:table_summary[k] for k in tables_select}
  
  schema = [
    f"""{k}: {v['description']}\n\n
    COLUMN DESCRIPTION:\n{json.dumps(v['columns'], indent=4)}"""
    for k, v in table_summary.items()
  ]

  state[f"{name}_schema"] = f"\n\n{'-'*80}\n\n".join(schema)
  state[f"{name}_dialect"] = db.dialect
  return state


def generate_query(state: Dict, config: Dict) -> Dict:
  """ Generates database query to answer user request"""
  name = config.get("name", "sqldb")
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
  invoke_state = {k.replace(f"{name}_", ""):v for k, v in state.items()}
  result = chain.invoke(invoke_state)
  
  # update state
  state[f"{name}_query"] = result
  state[f"{name}_ntry"] += 1
  return state


def run_query(state: Dict, config: Dict) -> Dict:
  """ Runs SQL query and returns result as pd.DataFrame """
  
  db = config.get("db")
  name = config.get("name", "sqldb")
  verbose = config.get("verbose", False)
  
  if verbose:
    print(" --- Checking query ---")
    
  # Check connection to database
  if db is None:
    raise Exception("No database spcified in config")
  
  # Extract sql query from code block
  block_query = state[f"{name}_query"] 
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
    state[f"{name}_messages"] = [AIMessage(result)]
  else:
    state[f"{name}_messages"] = [AIMessage("SUCCESS!")]
    state[f"{name}_df"] = result
    state[f"{name}_query"] = query
    
  return state


def route(state: Dict, config: Dict) -> Literal["debug", "__end__"]:
  """ Routes agent action to either debug or table summary """ 
  ntry = config.get("ntry", NTRY)
  name = config.get("name", "sqldb")

  if state.get(f"{name}_df") is None:
    return "summarize" if state[f"{name}_ntry"] > ntry else "debug"
  else:
    return "summarize"


def debug_query(state: Dict, config: Dict) -> Dict:
  """ Generates database query to answer user request"""
  name = config.get("name", "sqldb")
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
  invoke_state = {k.replace(f"{name}_", ""):v for k, v in state.items()}
  result = chain.invoke(invoke_state)
  
  # update state
  state[f"{name}_query"] = result
  state[f"{name}_ntry"] += 1
  return state


def summarize_table(state: Dict, config: Dict) -> Dict:
  """ Generates plain text description of table generated from SQL query"""
  
  llm = config.get("llm", LLM)
  name = config.get("name", "sqldb")
  verbose = config.get("verbose", False)  
  
  if verbose:
    print("--- Summarizing query ---")
    
  if state.get(f"{name}_query") is None:
    state[f"{name}_df_summary"] = "Did not generate query successfully"
  if state.get(f"{name}_schema") is None:
    state[f"{name}_df_summary"] = "Did not initialize schema successfully"

  prompt = PromptTemplate.from_template(template=SUMMARIZE_TABLE_PROMPT)
  chain = prompt | llm | StrOutputParser()

  # Clean state names for prompt invoking
  invoke_state = {k.replace(f"{name}_", ""):v for k, v in state.items()}
  invoke_state["table_string"] = state[f"{name}_df"].head().to_string()  
  state[f"{name}_df_summary"] = chain.invoke(invoke_state)
  
  return state  

