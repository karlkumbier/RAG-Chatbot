from typing import Dict, List
from langchain.utilities.sql_database import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage

from agent.sqldb.prompts import (
  SQL_SELECT_TABLE_PROMPT, SQL_QUERY_DESCRIPTION_PROMPT
)

import pandas as pd
import re
import json

def summarize_table_node_(state: Dict, llm: BaseLanguageModel, name: str) -> Dict:
  """ Generates plain text description of table generated from SQL query"""
  
  if state.get("db_query") is None:
    return "Query did not run"
  if state.get("db_schema") is None:
    raise Exception("schema not initialized")

  prompt = PromptTemplate.from_template(template=SQL_QUERY_DESCRIPTION_PROMPT)
  chain = prompt | llm | StrOutputParser()
  state["df_string"] = state["df"].head().to_string()
  state["df_summary"] = chain.invoke(state)
  return state  

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

  state['db_schema'] = f"\n\n{'-'*80}\n\n".join(schema)
  state['dialect'] = db.dialect
  return state


def make_table_list(db: SQLDatabase) -> List:
  """ Queries database to generate list of all all tables in database"""
  query = """SELECT
    table_name
    FROM
        information_schema.tables
    WHERE
        table_schema = 'public' -- Replace 'public' with your schema name
        AND table_type = 'BASE TABLE'
    ORDER BY
        table_name;
  """
  
  tables = pd.read_sql(query, db._engine) 
  return list(tables.table_name)


def get_column_comments(db: SQLDatabase, table: str) -> pd.DataFrame:
  """ Queries database to obtain column comments for a selected table. """
  query = f"""
    SELECT
        att.attname AS column_name,
        dsc.description AS column_comment
    FROM
        pg_attribute att
        JOIN pg_class cls ON att.attrelid = cls.oid
        LEFT JOIN pg_description dsc ON att.attrelid = dsc.objoid AND att.attnum = dsc.objsubid
    WHERE
        cls.relname = '{table}'
        AND att.attnum > 0
        AND NOT att.attisdropped
    ORDER BY
        att.attnum;
  """
  
  return pd.read_sql(query, db._engine) 


def get_table_comments(db: SQLDatabase, table: str) -> str:
  """ Queries database to obtain comments associated a table. """
  query = f"""
    SELECT
      cls.relname AS table_name,
      dsc.description AS table_comment
    FROM
      pg_class cls
      LEFT JOIN pg_description dsc ON cls.oid = dsc.objoid AND dsc.objsubid = 0
      LEFT JOIN pg_namespace nsp ON cls.relnamespace = nsp.oid
    WHERE
      cls.relname = '{table}'
      AND nsp.nspname = 'public';
  """
  
  result = pd.read_sql(query, db._engine)
  return str(result.table_comment[0])


def make_table_summary(table: str, table_comments: str, column_comments: pd.DataFrame, db: SQLDatabase) -> Dict:
  """ Generates a table summary dictionary with table description, column descriptions, and example rows
  """    
  
  out = {
    "description": table_comments,
    "columns":  dict(zip(
      column_comments.column_name, column_comments.column_comment
    )),
    "example": db.get_table_info_no_throw([table]) 
  }
  
  return out


def select_table(llm: BaseLanguageModel, question: str, summary: Dict) -> str:
  """ Uses llm to select a subset of tables from an sql database that are
  relevant to a user question. Selection is informed by table summaries generated with `make_table_summary`
  """
  prompt = PromptTemplate.from_template(template=SQL_SELECT_TABLE_PROMPT)
  chain = prompt | llm | StrOutputParser()

  tables = ", ".join(summary.keys())
  summary = [f"{k}: {v['description']}\n\n" for k, v in summary.items()]
  summary = "".join(summary)
  
  result = chain.invoke({
    "tables": tables,
    "summary": summary,
    "question": question
  })

  return result


def extract_sql_query(query: str) -> str:
  """ Extract formatted sql query. Query is assumed to be formatted as 
  ``` {query} ``` or ```sql {query ```}
  """  
  
  query = query.replace("```sql", "```")
  pattern = r'```\s(.*?)```'
  matches = re.findall(pattern, query, re.DOTALL)
  
  if not matches:
    return None
  else:
    return matches[0]