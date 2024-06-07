from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from typing import List
from langchain_core.retrievers import BaseRetriever

from agent.rag.nodes import *
from agent.models import retriever

class GraphState(TypedDict):
    question : str
    rag_response : str
    documents : List[str]
    retriever : BaseRetriever
    
# Define the nodes
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "__end__")
rag_agent = workflow.compile()
rag_agent.get_graph().print_ascii()

if __name__ == "__main__":
    question = """What are cancer persisters?"""

    result = rag_agent.invoke({
        "question":question, 
        "retriever":retriever
    })
    
    print(result.get("rag_response"))