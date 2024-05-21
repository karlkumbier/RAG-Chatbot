# Persister Information Center for AI-assisted Research and Development
from langgraph.graph import StateGraph

from typing_extensions import TypedDict
from typing import List
from chains import *
from utils import *

import pandas as pd

class GraphState(TypedDict):
    question : str
    generation : str
    code : List[str]
    documents : List[str]
    error_msg: List[str]
    attempts: int
    df: pd.DataFrame
    

def retrieve(state):
    """Retrieve documents from vectorstore."""
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state

def generate(state):
    """Generate answer using RAG on retrieved documents."""
    question = state["question"]
    docs = format_docs(state["documents"])
    
    # RAG generation
    generation = rag_chain.invoke({"context": docs, "question": question})
    state["generation"] = generation
    return state

def analyze(state):
    """Perform analysis to answer user question."""
    question = state["question"]
    
    # RAG generation
    if state.get("error_msg") is None:
        code = plot_chain.invoke({
            "question": question, 
            "column_names": state["df"].columns
        })

        code = extract_python_code(code)
        code = code.replace("fig.show()", "")
        state["code"] = [code]
        state["error_msg"] = []
        state["attempts"] = 1
        
    else:
        print("---DEBUGGING THE FOLLOWING ERROR---")
        print(state["error_msg"][-1])
        
        code = debug_chain.invoke({
            "code": state["code"], 
            "error_msg": state["error_msg"]
        })

        code = extract_python_code(code)
        code = code.replace("fig.show()", "")
        state["code"].append(code)
        state["attempts"] += 1
        
    
    
    try: 
        df = state["df"].copy()
        exec(code)
        state["error_msg"].append(None)
    except Exception as e:
        state["error_msg"].append(e)
    
    return state

def route_question(state):
    """Route question to web search or RAG."""

    question = state["question"]
    source = router_chain.invoke({"question": question})  
    
    if source == 'data_analysis':
        print("---ROUTING QUESTION TO ANALYSIS ENGINE---")
        return "data_analysis"
    else:
        print("---ROUTING QUESTION TO RAG---")
        return "text_summary"

def should_continue(state): 
    """ Determine whether to contine or not"""
    if state.get("error_msg")[-1] is None:
        return "__end__"
    elif state.get("attempts") > 10:
        return "__end__"
    else:
        print("---DEBUGGING---")
        return "analyze"
    

# Define the nodes
workflow = StateGraph(GraphState)
workflow.add_node("analyze", analyze)
workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(
    route_question,
    {
        "data_analysis": "analyze",
        "text_summary": "retrieve",
    },
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "__end__")
workflow.add_conditional_edges("analyze", should_continue)

agent = workflow.compile()