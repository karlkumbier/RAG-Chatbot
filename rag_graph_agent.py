# Persister Information Center for AI-assisted Research and Development
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnablePassthrough, 
    RunnableParallel, 
    RunnableLambda
)
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


import git
import os
import pandas as pd

from rag import format_docs
from operator import itemgetter

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

# Initialize embedder used to vectorize queries
embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-large-1",
)

# Load local vector database and initialize retriever for top-3 docs given query
faiss_db = os.environ["PICARD_DB"] 
index = FAISS.load_local(
    faiss_db, embedder, allow_dangerous_deserialization=True
)

retriever = index.as_retriever(search_kwargs={"k":5})

# Initialize chatbot LLM
gpt4 = AzureChatOpenAI(
    azure_deployment='gpt-4-turbo-128k',
    streaming=True
)

gpt35 = AzureChatOpenAI(
    azure_deployment='gpt-35-turbo-16k',
    temperature=0,
    streaming=True
)

RAG_LLM = gpt4
ROUTER_LLM = gpt35

###############################################################################
# Initialize agent for performing retrieval augmented generation
###############################################################################
#rag_base_prompt = """
#    Answer the question below using the provided context. The context provided are snippets from academic articles, with ARTICLE REFERENCE, formatted as AUTHOR (YEAR), indicated before each snippet. Cite any part of your answer that depends on a provided snippet using the ARTICLE REFERENCE associated with each snippet. Use inline citations formatted as (AUTHOR, YEAR)
#    
#    Do not use information outside of the provided context to answer the question.
#    
#    If the context is not relevant to the question, reply "I can't determine that from my available information":
#
#    Context: {context}
#
#    Question: {question}
#
#    Answer:
#"""
#
#rag_prompt = ChatPromptTemplate.from_messages(
#    [("system", rag_base_prompt), ("user", "{question}")]
#)

rag_prompt = PromptTemplate(template="""
    Answer the question below using the provided context. The context provided are snippets from academic articles, with ARTICLE REFERENCE, formatted as AUTHOR (YEAR), indicated before each snippet. Cite any part of your answer that depends on a provided snippet using the ARTICLE REFERENCE associated with each snippet. Use inline citations formatted as (AUTHOR, YEAR)
    
    Do not use information outside of the provided context to answer the question.
    
    If the context is not relevant to the question, reply "I can't determine that from my available information":

    Context: {context}

    Question: {question}

    Answer:
    """,
    input_variables=["question", "docs"]
)


rag_chain = rag_prompt | RAG_LLM | StrOutputParser()
#formatter = itemgetter("docs") | RunnableLambda(format_docs)


#rag_agent = (
#    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
#    .assign(context=formatter)
#    .assign(answer=answer)
#    .pick(["answer", "docs"])
#)

#answer, docs = rag_agent.invoke("what are cancer persisters?")

###############################################################################
# Initialize agent for working with pandas DataFrames
###############################################################################
x = pd.read_csv("sandbox/gene_de.csv").filter(
    ["SYMBOL", 
    "CellLine", 
    "pvalue", 
    "log2FoldChange", 
    "Contrast", 
    "ContrastFull"
    ]
)

data_agent = create_pandas_dataframe_agent(
    gpt35, 
    x, 
    verbose=True,
    return_intermediate_steps=True
)

###############################################################################
# Initialize routing chain
###############################################################################
router_template =  PromptTemplate.from_template(
    """You are an expert at routing user questions to either `data_analysis` or `text_summary`. 
    
    `data_analysis` questions typically ask to generate figures or compute summary statistics. For example, "generate a plot of log2 fold change versus p-value" or "how many genes have a p-value smaller than 0.05?". The typically involve queries around cell lines, conditions, p-values, and log2 fold change. You do not need to be explicit with these keywords.
    
    `text_summary` questions involve queries around more general biological processes.
    
    Classify each question as either 'data_analysis' or 'text_summary'. Do not respond with more than one word. 
    
    Question to route: {question}

    Classification:
    """
) 

router_chain = router_template | ROUTER_LLM | StrOutputParser()

def route(info):
    print(info)
    if "data_analysis" in info["topic"].lower():
        return data_agent
    else:
        return rag_chain
    
######### TESTING RAG graph
from typing_extensions import TypedDict
from typing import List

### State
class GraphState(TypedDict):
    question : str
    generation : str
    analysis : str
    documents : List[str]
    

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    docs = format_docs(state["documents"])
    
    # RAG generation
    generation = rag_chain.invoke({"context": docs, "question": question})
    state["generation"] = generation
    return state

def analyze(state):
    """
    Performs analysis to answer user question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM analysis
    """
    print("---Analyze---")
    question = state["question"]
    
    # RAG generation
    analysis = data_agent.invoke(question)
    state["analysis"] = analysis
    return state

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = router_chain.invoke({"question": question})  
    print(source)
    
    if source == 'data_analysis':
        print("---ROUTE QUESTION TO ANALYSIS ENGINE---")
        return "data_analysis"
    else:
        print("---ROUTE QUESTION TO RAG---")
        return "text_summary"
    

from langgraph.graph import END, StateGraph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("analyze", analyze) # web search
workflow.add_node("retrieve", retrieve) # retrieve
workflow.add_node("generate", generate) # generatae

workflow.set_conditional_entry_point(
    route_question,
    {
        "data_analysis": "analyze",
        "text_summary": "retrieve",
    },
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("analyze", END)

app = workflow.compile()
print(app.get_graph().draw_ascii())

inputs = {"question": "What are cancer persisters?"}

for output in app.stream(inputs):
    for key, value in output.items():
        print(f"Finished running: {key}:")

print(value["generation"])

result = app.invoke({"question": "Plot -log10 p-value versus log2 fold change. Generate a distinct plot for each cell line"})