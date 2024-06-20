from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

from agent.base_agent.agent import BaseAgent
from agent.sqldb import nodes
from typing_extensions import TypedDict
from typing import Sequence, Annotated, Dict

class SQLDBAgentState(TypedDict):
  question: str # user question
  messages: Annotated[Sequence[BaseMessage], add_messages] # chat history
  dialect: str # SQL dialect
  dbschema: str # comments on tables + columns
  results: Dict  # agent ouput
  ntry: int # number of attempts


class SQLDBAgent(BaseAgent):
  
  def __init__(self, name="sqldb"):
    super().__init__()
    self.agent = self.__build_graph__()
    self.name = name
    
  def __build_graph__(self): 
    """ Initializes logic flow of agent graph """
    workflow = StateGraph(SQLDBAgentState)
    workflow.add_node("initializer", nodes.initialize)
    workflow.add_node("query_generator", nodes.generate_query)
    workflow.add_node("query_runner", nodes.run_query)
    workflow.add_node("query_debugger", nodes.debug_query)
    workflow.add_node("table_summarizer", nodes.summarize_table)

    workflow.set_entry_point("initializer")
    workflow.add_edge("initializer", "query_generator")
    workflow.add_edge("query_generator", "query_runner")
    workflow.add_edge("query_debugger", "query_runner")
    workflow.add_edge("table_summarizer", "__end__")

    workflow.add_conditional_edges(
      "query_runner",
      nodes.route,
      {
        "debug": "query_debugger", 
        "summarize": "table_summarizer", 
        "__end__": "__end__"
      },
    )
    
    return workflow.compile()

if __name__ == "__main__":
  from agent.sqldb.agent import SQLDBAgent
  from agent.models import gpt4
  from langchain.utilities.sql_database import SQLDatabase

  # Initialize connection to database 
  username = "picard"
  password = "persisters"
  port = "5432"
  host = "localhost"
  dbname = "persisters"
  pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"
  db = SQLDatabase.from_uri(pg_uri)
  
  # Initialize agent
  sqldb_agent = SQLDBAgent()
  sqldb_agent.__print__()
  config = {"db": db, "llm": gpt4, "name": sqldb_agent.name, "verbose": True}

  # Test 1:
  question = """
    Get a tables of log2 fold change, p-value, and gene name from the hypoxia 
    vs. normoxia screen. Filter these to PC9 cell lines in the baseline
    context. Limit results to 20 samples
  """
  
  sqldb_agent.__set_state__({"question": question}, config)  
  print(sqldb_agent.get("results").get("summary"))
  sqldb_agent.get("results").get("df") 
  
  # Test 2:
  question = """
    Generate a table of differential expression results that includes data on 
    all available cell lines. Include all available columns. Use only the 
    baseline context for each cell line and the standard of care versus no drug 
    contrast. Return all rows from this table.  
  """
  
  sqldb_agent.__set_state__({"question": question}, config)
  print(sqldb_agent.get("results").get("summary"))
  sqldb_agent.get("results").get("df") 