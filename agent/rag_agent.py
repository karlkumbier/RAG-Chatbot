from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from typing import List, Sequence, Annotated, Dict
from langchain_core.messages import BaseMessage

from agent.rag.utils import *
from agent.rag.prompts import RAG_PROMPT
from agent.models import retriever, gpt4
from agent.base_agents import *

import functools
import operator

LLM = gpt4

# Set agent initialer and router
def initializer_node(state: Dict) -> Dict:
  """ Initialize messages"""
  if state.get("messages") is None:
    state["messages"] = []
    
  return state
  

# Node for tool calling agent - generates tool inputs
retriever_node = functools.partial(
  retriever_node_,
  retriever=retriever
)

generator_node = functools.partial(
  agent_node, 
  agent=chat_agent(LLM, RAG_PROMPT), 
  name="generator_node"
)

###############################################################################
# Initialize agent graph 
###############################################################################
class GraphState(TypedDict):
  question : str
  messages : Annotated[Sequence[BaseMessage], operator.add]
  context : str
  documents : List[str]

# Define the nodes
workflow = StateGraph(GraphState)
workflow.add_node("initializer", initializer_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("generator", generator_node)

workflow.set_entry_point("initializer")
workflow.add_edge("initializer", "retriever")
workflow.add_edge("retriever", "generator")
workflow.add_edge("generator", "__end__")
rag_agent = workflow.compile()
#rag_agent.get_graph().print_ascii()

if __name__ == "__main__":
  question = """What are cancer persisters?"""
  result = rag_agent.invoke({"question":question})
  print(result.get("messages")[-1].content)