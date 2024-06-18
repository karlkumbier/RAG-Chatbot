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

  # Check connection to database
  if config.get("db") is None:
    raise Exception("No database specified in config")
  
  if state.get("messages") is None:
    state["messages"] = []
    
  if state.get("ntry") is None:
    state["ntry"] = 0
    
  return state


def set_schema(state: Dict, config: Dict) -> Dict:
  """ Queries database and initializes graph state schema. """ 
  db = config.get("db")
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
  tables_select = select_table(llm, state["question"], table_summary)
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

  state["schema"] = f"\n\n{'-'*80}\n\n".join(schema)
  state["dialect"] = db.dialect
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
  state["query"] = result
  state["ntry"] += 1
  return state


def run_query(state: Dict, config: Dict) -> Dict:
  """ Runs SQL query and returns result as pd.DataFrame """
  db = config.get("db")
  verbose = config.get("verbose", False)
  
  if verbose:
    print(" --- Checking query ---")
    
  # Check connection to database
  if db is None:
    raise Exception("No database spcified in config")
  
  # Extract sql query from code block
  block_query = state["query"] 
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
  else:
    state["messages"] = [AIMessage("SUCCESS!")]
    state["df"] = result
    state["query"] = query
    
  return state


def route(state: Dict, config: Dict) -> Literal["debug", "__end__"]:
  """ Routes agent action to either debug or table summary """ 
  ntry = config.get("ntry", NTRY)

  if state.get("df") is None:
    return "summarize" if state["ntry"] > ntry else "debug"
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
  result = chain.invoke(state)
  
  # update state
  state["query"] = result
  state["ntry"] += 1
  return state


def summarize_table(state: Dict, config: Dict) -> Dict:
  """ Generates plain text description of table generated from SQL query"""
  
  llm = config.get("llm", LLM)
  verbose = config.get("verbose", False)  
  
  if verbose:
    print("--- Summarizing query ---")
    
  if state.get("query") is None:
    state["df_summary"] = "Did not generate query successfully"
  if state.get("schema") is None:
    state["df_summary"] = "Did not initialize schema successfully"

  prompt = PromptTemplate.from_template(template=SUMMARIZE_TABLE_PROMPT)
  chain = prompt | llm | StrOutputParser()

  # Clean state names for prompt invoking
  state["table_string"] = state["df"].head().to_string()  
  state["df_summary"] = chain.invoke(state)
  
  return state  

