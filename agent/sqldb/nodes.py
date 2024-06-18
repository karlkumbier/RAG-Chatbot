from typing import Dict, Literal
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from agent.models import cold_gpt35
from agent.sqldb.utils import *
from agent.sqldb.prompts import *
from agent.base_agents import agent_node, chat_agent

import functools
import pandas as pd
import json
LLM = cold_gpt35
NTRY = 10

generate_query = functools.partial(
  agent_node, 
  agent=chat_agent(LLM, GENERATE_QUERY_PROMPT), 
  name="query_generator"
)

debug_query = functools.partial(
  agent_node,
  agent=chat_agent(LLM, DEBUG_QUERY_PROMPT),
  name="query_debugger"
)

def initialize(state: Dict, config: Dict) -> Dict:
  """ Initialize baseline state variables """
  name = config.get("name", "sqldb")

  if state.get(f"{name}_messages") is None:
    state[f"{name}_messages"] = []
    
  if state.get(f"{name}_ntry") is None:
    state[f"{name}_ntry"] = 0
    
  return state


def route(state: Dict, config: Dict) -> Literal["debug", "__end__"]:
  """ Routes agent action to either debug or table summary """ 
  ntry = config.get(ntry, NTRY)
  name = config.get("name", "sqldb")

  if state.get(f"{name}_df") is None:
    return "summarize" if state["ntry"] > ntry else "debug"
  else:
    return "summarize"


def summarize_table(state: Dict, config: Dict) -> Dict:
  """ Generates plain text description of table generated from SQL query"""
  
  llm = config.get("llm", LLM)
  name = config.get("name", "sqldb")
  
  if state.get(f"{name}_query") is None:
    state[f"{name}_df_summary"] = "Did not generate query successfully"
  if state.get(f"{name}_schema") is None:
    state[f"{name}_df_summary"] = "Did not initialize schema successfully"

  prompt = PromptTemplate.from_template(template=SUMMARIZE_TABLE_PROMPT)
  chain = prompt | llm | StrOutputParser()

  # Clean state names for prompt invoking
  invoke_state = {k.replace(f"{name}_", ""):v for k, v in state.items()}
  invoke_state["df_string"] = state["df"].head().to_string()  
  state[f"{name}_df_summary"] = chain.invoke(invoke_state)
  
  return state  


def run_query(state: Dict, config: Dict) -> Dict:
  """ Runs SQL query and returns result as pd.DataFrame """
  
  db = config.get("db")
  name = config.get("name", "sqldb")

  # Check connection to database
  if db is None:
    raise Exception("No database spcified in config")
  
  # Extract sql query from message thread
  query = state[f"{name}_messages"][-1].content
  query = extract_sql_query(query)
  
  # Try to run extracted code, raise exceptions if unsuccessful
  if query is None:
    result = SyntaxError(
      "Perhaps you forgot to wrap your query in ```. Example format:      ```query```"
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
    state[f"{name}_query"] = query
    state[f"{name}_df"] = result
    
  return state


def set_schema(state: Dict, config: Dict) -> Dict:
  """ Queries database and initializes graph state schema. """
  
  db = config.get("db")
  name = config.get("name", "sqldb")
  llm = config.get("llm", LLM)

  # Check connection to database
  if db is None:
    raise Exception("No database spcified in config")

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

  state[f"{name}_schema"] = f"\n\n{'-'*80}\n\n".join(schema)
  state[f"{name}_dialect"] = db.dialect
  return state