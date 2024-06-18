from agent.base_agents import chat_agent
from agent.models import cold_gpt35, gpt4
from agent.geordi.prompts import *
from langgraph.graph import StateGraph
from langchain_core.language_models import BaseLanguageModel

from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from typing import Sequence, Annotated, Literal, Dict
from typing_extensions import TypedDict
import operator

LLM = gpt4

question = """Generate a figure of differential expression p-value versus log2 
fold change for PC9 cell lines. The differential expression analysis should
contrast SOC with no drug. Use data from the hypoxia vs. normoxia screen. 
"""

# Initialize SQLDB base agent
from agent.sqldb.agent import SQLDBAgent
from agent.chart.agent import ChartAgent

from langchain.utilities.sql_database import SQLDatabase

username = "picard"
password = "persisters"
port = "5432"
host = "localhost"
dbname = "persisters"
pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{dbname}"
db = SQLDatabase.from_uri(pg_uri)

# TODO: data_loader should incorporate message from insufficiency explainer
# TODO: Fig explanation of generated figures
# TODO clean up

##############################################################################
# Agent assesses relevance of available data for downstream analysis. If data 
# sufficient, send on for processing.
##############################################################################
# Utility functions
def check_config(config: Dict) -> None:
  """ Runs checks on config """
  
  # Ensure databse connection in config
  if config.get("db") is None:
    raise Exception("Must specify database `db` in `config`")
  
  if not isinstance(config.get("db"), SQLDatabase):
    raise TypeError("`db` must be an SQLDatabase")

  return None


def check_relevance(question: str, agent: SQLDBAgent, llm: BaseLanguageModel) -> Literal["sufficient", "insufficient"]:
  """ Determines whether data are sufficient or insufficient for request"""
  chain = chat_agent(llm, RELEVANCE_PROMPT)
  
  summary = agent.get("df_summary")
  invoke_state = {"df_summary": summary, "question": question}
  return chain.invoke(invoke_state).content


# Define agent state
class AgentState(TypedDict):
  question: str # user question
  messages: Annotated[Sequence[BaseMessage], operator.add] # chat history
  sqldb_agent: SQLDBAgent
  chart_agent: ChartAgent
  ntry: int # number of debug attemps


# Define graph node functions
def initialize(state: Dict, config: Dict) -> Dict:
  """ Initialize graph state values """
  
  # Initialize state values
  check_config(config)
  
  if state.get("sqldb_agent") is None:
    state["sqldb_agent"] = SQLDBAgent() 

  if state.get("chart_agent") is None:
    state["chart_agent"] = ChartAgent()
    
  if state.get("messages") is None:
    state["messages"] = []
    
  if state.get("ntry") is None:
    state["ntry"] = 0
    
  return state


def load_data(state: Dict, config: Dict) -> Dict:
  """ Calls sqldb agent to load data """
  check_config(config)
  sqldb_agent = state["sqldb_agent"] 

  if config.get("verbose", False):
    print("--- LOADING DATASET ---")

  # Run sql agent to load data
  invoke_state = {"question": state.get("question")}
  sqldb_agent.state = sqldb_agent.invoke(invoke_state, config)   
  state["sqldb_agent"] = sqldb_agent
  state["ntry"] += 1

  # Add messages from sql agent to chat thread
  if sqldb_agent.get("df") is not None:
    summary = sqldb_agent.get("df_summary")
    msg = f"Successfully loaded the following table: {summary}"
    state["messages"] = AIMessage(msg, name=sqldb_agent.name)
  else:
    msg = f"Failed to load table."
    state["messages"] = AIMessage(msg, name=sqldb_agent.name)
    
  return state


def router(state: Dict, config: Dict) -> Literal["figure", "analysis"]:
  """ Determine which downstream task to be performed"""
  llm = config.get("llm", LLM)
  sqldb_agent = state["sqldb_agent"] 
  question = state.get("question")  
  relevance = check_relevance(question, sqldb_agent, llm)
  
  if config.get("verbose", False):
    print("--- ROUTING ---")
    print(relevance)

  # If data cannot address request, send back to explainer
  if relevance == "insufficient":
    return "insufficiency_explainer"
  
  chain = chat_agent(llm, TASK_PROMPT)
  invoke_state = {"question": question}
  return chain.invoke(invoke_state).content 


def explain_insufficient(state: Dict, config: Dict) -> Dict:
  """ Provide a natural language description of why data are insufficient"""
  llm = config.get("llm", LLM)
  sqldb_agent = state["sqldb_agent"] 
  
  chain = chat_agent(llm, EXPLAIN_IRRELEVANCE_PROMPT)
  
  invoke_state = {
    "question": state["question"],
    "table": sqldb_agent.get("df").head().to_string(),
    "df_summary": sqldb_agent.get("df_summary"),
    "query": sqldb_agent.get("query")
  }

  result = StrOutputParser(chain.invoke(invoke_state))
  state["messages"] = [AIMessage(result)]
  return state 
  

def make_figure(state: Dict, config: Dict) -> Dict:
  """ Calls chart agent to generate a figure """
  chart_agent = state["chart_agent"]  
  sqldb_agent = state["sqldb_agent"] 

  if config.get("verbose", False):
    print("--- GENERATING FIGURE ---")
  
  config["df"] = sqldb_agent.get("df")
  invoke_state = {"question": state["question"]}
  chart_agent.state = chart_agent.invoke(invoke_state, config=config)
  state["chart_agent"] = chart_agent
  return state

def run_analysis(state: Dict)  -> Dict:
  """ Calls analysis agent to run analysis"""
  message = "Analysis engine not curently supported"
  state["messages"] = [AIMessage(message)]
  return state

# Construct agent graph
workflow = StateGraph(AgentState)
workflow.add_node("initializer", initialize)
workflow.add_node("data_loader", load_data)
workflow.add_node("insufficiency_explainer", explain_insufficient)
workflow.add_node("figure_generator", make_figure)
workflow.add_node("analysis_runner", run_analysis)

workflow.set_entry_point("initializer")
workflow.add_edge("initializer", "data_loader")
workflow.add_edge("insufficiency_explainer", "data_loader")
workflow.add_edge("figure_generator", "__end__")
workflow.add_edge("analysis_runner", "__end__")

workflow.add_conditional_edges(
  "data_loader",
  router,
  {
    "insufficiency_explainer": "insufficiency_explainer", 
    "figure": "figure_generator", 
    "analysis": "analysis_runner"
  }
)

geordi = workflow.compile()
geordi.get_graph().print_ascii()

config = {"verbose": True, "db": db, "llm": LLM, "name": "geordi"}
result = geordi.invoke({"question": question}, config=config)



##############################################################################
# If data are not relevant, describe problem, retry query
##############################################################################

##############################################################################
# Once data are relevant...
##############################################################################

##############################################################################
# Determine task to be performed - task should be in state, to be read by 
# supervising agents
##############################################################################

##############################################################################
# Carry out task
# 1. Update figure agent according to state / table summary / new changes
# 2. Add node for `analysis` to return message: methods not implemented yet
# 3. Modify state (e.g., by adding fig). Add state message summarizing the fig.
##############################################################################

##############################################################################
# Assess task completeness - chat history / single message node
##############################################################################

##############################################################################
# If task not complete, describe problem, route description to one of database # loader or figure generator and rerun cycle
##############################################################################

