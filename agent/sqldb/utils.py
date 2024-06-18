from typing import Dict, List
from langchain.utilities.sql_database import SQLDatabase
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from agent.sqldb.prompts import SELECT_TABLE_PROMPT

import pandas as pd
import re

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
  """ Queries database to obtain comments associated with a table. """
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
  """ Generates a table summary including: table description, column 
  descriptions, and example rows from table
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
  prompt = PromptTemplate.from_template(template=SELECT_TABLE_PROMPT)
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