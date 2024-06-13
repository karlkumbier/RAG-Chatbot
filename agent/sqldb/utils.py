from typing import Dict, List
from langchain.utilities.sql_database import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from agent.sqldb.prompts import SQL_SELECT_TABLE_PROMPT

import pandas as pd
import re

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


def set_schema_node_new_(state: Dict, llm: BaseLanguageModel, db: SQLDatabase) -> Dict:
  """ Queries database and initializes graph state schema.
  
  Graph `state` must include `schema` entry to store results
  """
  
  schema_query = """
    SELECT oid, relname, description FROM pg_class
    INNER JOIN pg_description ON pg_class.oid = pg_description.objoid;
  """
  
  st = pd.read_sql(schema_query, db._engine)
  table_schemas = {}
  
  for tbl in set(st.relname):
    column_query = f"""SELECT * FROM "{tbl}" LIMIT 0;"""
    columns = list(pd.read_sql(column_query, db._engine).columns)
    description = list(st[st.relname == tbl].description)
    
    table_schemas[tbl] = {
      "description": description[0],
      "columns":  dict(zip(columns, description[1:])),
      "example": db.get_table_info_no_throw([tbl]) 
    }
    
  
  return None
  

def set_schema_node_(state: Dict, llm: BaseLanguageModel, db: SQLDatabase) -> Dict:
  """ Queries database and initializes graph state schema.
  
  Graph `state` must include `schema` entry to store results
  """
  
  tables = db.get_usable_table_names()
  tables_select = select_table_agent(llm, state["question"], tables)
  tables_select = ",".split(tables_select)
  
  state["db_schema"] = db.get_table_info_no_throw(tables_select)
  state["dialect"] = db.dialect
  return state
  
  
def select_table_agent(llm: BaseLanguageModel, query: str, tables: List[str]) -> str:
  """ Determines which table(s) are relevant to the user question"""
  prompt = PromptTemplate.from_template(template=SQL_SELECT_TABLE_PROMPT)
  chain = prompt | llm | StrOutputParser()
  
  result = chain.invoke({
    "tables": ", ".join(tables),
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