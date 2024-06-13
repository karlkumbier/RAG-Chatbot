from typing import Dict
from langchain.utilities.sql_database import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from agent.sqldb.prompts import SQL_SELECT_TABLE_PROMPT

import pandas as pd
import re
import json

def run_query_node_(state: Dict, db: SQLDatabase, name: str) -> Dict:
  """ Runs SQL query and returns result as pd.DataFrame
  
  Graph `state` must include `sql_query` and `df` entries
  """
  query = state["messages"][-1].content
  query = extract_sql_query(query)
  
    # Check that code was properly extracted
  if query is None:
    result = SyntaxError(
      "Perhaps you forgot to wrap your query in ```. Example format: ```query```"
    )
    
    return result
  
  try:
    result = pd.read_sql(query, db._engine)
  except BaseException as e:
    result = e
    
  if isinstance(result, Exception):
    result = f"Failed to execute QUERY:\n\n{query}.\n\nERROR: {repr(result)}"
    state["messages"] = [AIMessage(result, name=name)]
  else:
    state["db_query"] = query
    state["df"] = result
    
  return state


def set_schema_node_(state: Dict, llm: BaseLanguageModel, db: SQLDatabase) -> Dict:
  """ Queries database and initializes graph state schema.
  
  Graph `state` must include `schema` entry to store results
  """
  
  # Query SQL database for description table: comments on tables/cols
  schema_query = """
    SELECT oid, relname, description FROM pg_class
    INNER JOIN pg_description ON pg_class.oid = pg_description.objoid;
  """
  
  dt = pd.read_sql(schema_query, db._engine)
  tables = set(dt.relname)
  table_summary = {tbl:make_table_summary(tbl, dt, db) for tbl in tables}
  
  # Select tables that are relevant to the query
  tables_select = select_table_agent(llm, state["question"], table_summary)
  tables_select = tables_select.split(", ")
  
  if tables_select is "None":
    raise Exception("Tables relevant to the question cannot be found")

  # Initialize schema for relevant tables
  table_summary = {k:table_summary[k] for k in tables_select}
  
  schema = [
    f"""{k}: {v['description']}\n\n
    COLUMN DESCRIPTION:\n{json.dumps(v['columns'], indent=4)}"""
    for k, v in table_summary.items()
  ]

  state['db_schema'] = f"\n\n{'-'*80}\n\n".join(schema)
  state['dialect'] = db.dialect
  return state
  

def make_table_summary(table: str, dt: pd.DataFrame, db: SQLDatabase) -> Dict:
  """ Generates a table summary dictionary with table description, column descriptions, and example rows
  """
  column_query = f"""SELECT * FROM "{table}" LIMIT 0;"""
  columns = list(pd.read_sql(column_query, db._engine).columns)
  description = list(dt[dt.relname == table].description)
    
  out = {
    "description": description[0],
    "columns":  dict(zip(columns, description[1:])),
    "example": db.get_table_info_no_throw([table]) 
  }
  
  return out
  
def set_schema_node_old(state: Dict, llm: BaseLanguageModel, db: SQLDatabase) -> Dict:
  """ Queries database and initializes graph state schema.
  
  Graph `state` must include `schema` entry to store results
  """
  
  tables = db.get_usable_table_names()
  tables_select = select_table_agent(llm, state["question"], tables)
  tables_select = ",".split(tables_select)
  
  state["db_schema"] = db.get_table_info_no_throw(tables_select)
  state["dialect"] = db.dialect
  return state
  
  
def select_table_agent(llm: BaseLanguageModel, query: str, table_summary: Dict) -> str:
  """ Determines which table(s) are relevant to the user question"""
  prompt = PromptTemplate.from_template(template=SQL_SELECT_TABLE_PROMPT)
  chain = prompt | llm | StrOutputParser()

  tables = ", ".join(table_summary.keys())
  summary = [f"{k}: {v['description']}\n\n" for k, v in table_summary.items()]
  summary = "".join(summary)
  
  result = chain.invoke({
    "tables": tables,
    "summary": summary,
    "question": query
  })

  return result


def extract_sql_query(query: str) -> str:
  """Extract formatted sql query"""  
  query = query.replace("```sql", "```")
  pattern = r'```\s(.*?)```'
  matches = re.findall(pattern, query, re.DOTALL)
  if not matches:
    return None
  else:
    return matches[0]