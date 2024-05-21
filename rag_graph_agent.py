# Persister Information Center for AI-assisted Research and Development
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.faiss import FAISS

from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, StateGraph

from typing_extensions import TypedDict
from typing import List

import git
import os
import re
import pandas as pd
from rag import format_docs

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
PLOT_LLM = gpt35

def extract_python_code(text):
    pattern = r'```python\s(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        return None
    else:
        return matches[0]

###############################################################################
# Initialize agent for performing retrieval augmented generation
###############################################################################
rag_prompt = PromptTemplate.from_template(template="""
    Answer the question below using the provided context. The context provided are snippets from academic articles. An ARTICLE REFERENCE, formatted as AUTHOR (YEAR), is indicated before each snippet. Cite any part of your answer that depends on a provided snippet using the ARTICLE REFERENCE associated with each snippet. Use inline citations formatted as (AUTHOR, YEAR).
    
    Do not use information outside of the provided context to answer the question.
    
    If the context is not relevant to the question, reply "I can't determine that from my available information":

    Context: {context}

    Question: {question}

    Answer:
    """
)

rag_chain = rag_prompt | RAG_LLM | StrOutputParser()

###############################################################################
# Initialize agent for generating figures from pandas DataFrames
###############################################################################
df = pd.read_csv("sandbox/gene_de.csv").filter(
    ["SYMBOL", 
    "CellLine", 
    "pvalue", 
    "log2FoldChange", 
    "Contrast", 
    "ContrastFull"
    ]
)

plot_base_template = """
    The dataset is ALREADY loaded into a DataFrame named 'df'. DO NOT load the data again.
    
    The DataFrame has the following columns: {column_names}
    
    Before plotting, ensure the data is ready:
    1. Check if columns that are supposed to be numeric are recognized as such. If not, attempt to convert them.
    2. Handle NaN values by filling with mean or median.
    
    Provide SINGLE CODE BLOCK with a solution using Pandas and Plotly plots in a single figure to address the following query:
    
    {question}

    - USE SINGLE CODE BLOCK with a solution. 
    - Do NOT EXPLAIN the code 
    - DO NOT COMMENT the code. 
    - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK.
    - The code block must start and end with ```
    
    - Example code format ```code```

    - Colors to use for background and axes of the figure : #F0F0F6
"""

plot_prompt = ChatPromptTemplate.from_messages(
    [("system", plot_base_template), ("user", "{question}")]
)

plot_chain = plot_prompt | PLOT_LLM | StrOutputParser()


###############################################################################
# Initialize agent for debugging code
###############################################################################
debug_base_template = """
    Analyze the following CODE BLOCK and ERROR MESSAGE to diagnose the problem 
    with the code. Propose a solution that will resolve the error message.
    Provide NEW CODE BLOCK with a solution. Provide any explanation for your prosed solution after the NEW CODE BLOCK.      
    
    - The NEW CODE BLOCK must start and end with ``` 
    - Example NEW CODE BLOCK format ```code```
    
    
    CODE BLOCK:
    
    {code}
    
    ERROR MESSAGE:
    
    {error_msg}
    
    NEW CODE BLOCK:
"""

debug_prompt = ChatPromptTemplate.from_messages(
    [("system", debug_base_template)]
)

debug_chain = debug_prompt | PLOT_LLM | StrOutputParser()

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

###############################################################################
# Initialize agent graph
###############################################################################
class GraphState(TypedDict):
    question : str
    generation : str
    code : str
    documents : List[str]
    error_msg: str
    attempts: int
    

def retrieve(state):
    """Retrieve documents from vectorstore."""
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    state["documents"] = documents
    return state

def generate(state):
    """Generate answer using RAG on retrieved documents."""
    print("---GENERATE---")
    question = state["question"]
    docs = format_docs(state["documents"])
    
    # RAG generation
    generation = rag_chain.invoke({"context": docs, "question": question})
    state["generation"] = generation
    return state

def analyze(state):
    """Perform analysis to answer user question."""
    print("---Analyze---")
    question = state["question"]
    
    # RAG generation
    if state["error_message"] is None:
        code = plot_chain.invoke({
            "question": question, 
            "column_names": df.columns
        })

        code = extract_python_code(code)
        code = code.replace("fig.show()", "")
        state["code"] = code
        state["attempts"] = 1
        
    else:
        code = debug_chain.invoke({
            "code": state["code"], 
            "error_message": state["error_message"]
        })

        code = extract_python_code(code)
        code = code.replace("fig.show()", "")
        state["code"] = code  
        state["attempts"] += 1
    
    try:
        exec(code)
        state["error_msg"] = None
    except Exception as e:
        state["error_msg"] = e
    
    return state

def route_question(state):
    """Route question to web search or RAG."""

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = router_chain.invoke({"question": question})  
    
    if source == 'data_analysis':
        print("---ROUTING QUESTION TO ANALYSIS ENGINE---")
        return "data_analysis"
    else:
        print("---ROUTING QUESTION TO RAG---")
        return "text_summary"

def debug(state): 
    """ Attempt to debug code if error message generated"""
    if state["error_msg"] is None:
        return "complete"
    elif state["attempts"] > 5:
        return "max_attempts"
    else:
        print("---DEBUGGING---")
        return "debug"
    

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
workflow.add_edge("generate", END)

workflow.add_conditional_edges(
    "analyze",
    debug,
    {
        "complete": END,
        "max_attempts": END,
        "debug":"analyze"
    }
)

app = workflow.compile()
print(app.get_graph().draw_ascii())


result_rag = app.invoke({"question": "What are cancer persisters?"})
result_rag.get("generation")

result_analysis = app.invoke({"question": "Plot pvalue versus log2FoldChange"}) 
result_analysis.get("analysis")

result_analysis = app.invoke({
    "question": "Plot -log10 pvalue versus log2FoldChange"
}) 

exec(result_analysis.get("analysis"))
result_analysis.get("error_msg")

result_debug = debug_chain.invoke({
    "question":"Plot -log10 pvalue versus log2FoldChange",
    "error_msg": result_analysis.get("error_msg"),
    "column_names": df.columns,
    "code": result_analysis.get("analysis")
})


# loop
try:
    exec(extract_python_code(result_debug))
except Exception as e:
    error_msg = e
    
result_debug = debug_chain.invoke({
    "error_msg": error_msg,
    "code": extract_python_code(result_debug)
})

exec(extract_python_code(result_debug))
print(result_debug)
