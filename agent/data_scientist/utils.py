from agent.data_scientist.prompts import *
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.utilities.sql_database import SQLDatabase
from typing import Literal, Dict

# Utility functions
def check_config(config: Dict) -> None:
  """ Runs checks on config """
  
  # Ensure databse connection in config
  if config.get("db") is None:
    raise Exception("Must specify database `db` in `config`")
  
  if not isinstance(config.get("db"), SQLDatabase):
    raise TypeError("`db` must be an SQLDatabase")

  return None


def check_relevance(question: str, state: Dict, llm: BaseLanguageModel) -> Literal["sufficient", "insufficient"]:
  """ Determines whether data are sufficient or insufficient for request"""
  prompt = PromptTemplate.from_template(RELEVANCE_PROMPT)
  chain = prompt | llm | StrOutputParser()
  
  _state = {
    "question": question,
    "schema": state.get("schema"),
    "query": state.get("results").get("query"),
    "table": state.get("results").get("df").head().to_string()
  }
  
  return chain.invoke(_state)
