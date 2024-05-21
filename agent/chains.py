# Persister Information Center for AI-assisted Research and Development
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from models import *

###############################################################################
# Initialize chain for generating response from context
###############################################################################
rag_prompt = PromptTemplate.from_template(template="""
    You are an expert scientific research assistant. Answer the question below
    using the provided context. The provided context are snippets from academic
    articles. An ARTICLE REFERENCE, formatted as AUTHOR (YEAR), is indicated
    before each snippet. Cite any part of your answer that depends on a provided
    snippet using the ARTICLE REFERENCE associated with each snippet. Use inline
    citations formatted as (AUTHOR, YEAR).
    
    Do not use information outside of the provided context to answer the question.
    
    If the context is not relevant to the question, reply "I can't determine that from my available information":

    Context: {context}

    Question: {question}

    Answer:
    """
)

rag_chain = rag_prompt | RAG_LLM | StrOutputParser()

###############################################################################
# Initialize chain for generating figures from pandas DataFrames
###############################################################################
plot_base_template = """
    You are an expert data scientist. Generate a SINGLE CODE BLOCK that generates a figure to address the query below. Use Pandas and Plotly plots to produce the figure. The dataset is ALREADY loaded into a DataFrame named 'df'. DO NOT load the data again.
    
    The DataFrame has the following columns: {column_names}
    
    Before plotting, ensure that data for any columns used in your solution is ready:
    1. Drop any rows with NaN values in any column that will be used in the solution.
    2. Check if columns that are supposed to be numeric are recognized as such. If not, attempt to convert them.
    3. Perform any transformation of the data using numpy and remove infinite values.
    

    QUERY: {question}

    - USE SINGLE CODE BLOCK with a solution. 
    - Do NOT EXPLAIN the code 
    - DO NOT COMMENT the code. 
    - ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK.
    - The code block must start and end with ```
    
    - Example code format ```code```
"""

plot_prompt = ChatPromptTemplate.from_messages(
    [("system", plot_base_template), ("user", "{question}")]
)

plot_chain = plot_prompt | PLOT_LLM | StrOutputParser()

###############################################################################
# Initialize chain for debugging LLM-generated code
###############################################################################
debug_prompt = PromptTemplate.from_template(template="""
    You are an expert software engineer. Analyze the following CODE BLOCK and
    ERROR MESSAGE to diagnose the problem with the code. Propose a solution that
    will resolve the error message.  Provide NEW CODE BLOCK that implements the original CODE block along with the proposed solution. Provide any explanation for your prosed solution after the NEW CODE BLOCK.
    
    - The NEW CODE BLOCK must start and end with ```
    - The NEW CODE BLOCK should seek to perform the same function as the original CODE BLOCK while resolving any errors. 
    - Example NEW CODE BLOCK format ```code```
    
    
    CODE BLOCK:
    
    {code}
    
    ERROR MESSAGE:
    
    {error_msg}
    
    NEW CODE BLOCK:
"""
)

debug_chain = debug_prompt | PLOT_LLM | StrOutputParser()

###############################################################################
# Initialize chain for routing initial queries to either RAG or plotting
###############################################################################
router_template =  PromptTemplate.from_template(template="""
    You are an expert at routing user questions to either `data_analysis` or `text_summary`. 

    `data_analysis` questions typically ask to generate figures or compute summary statistics. For example, "generate a plot of log2 fold change versus p-value" or "how many genes have a p-value smaller than 0.05?". The typically involve queries around cell lines, conditions, p-values, and log2 fold change. You do not need to be explicit with these keywords.
    
    `text_summary` questions involve queries around more general biological processes.
    
    Classify each question as either 'data_analysis' or 'text_summary'. Do not respond with more than one word. 
    
    Question to route: {question}

    Classification:
    """
) 

router_chain = router_template | ROUTER_LLM | StrOutputParser()