# Persister Information Center for AI-assisted Research and Development
import os
import databutton as db
from typing import Tuple, List
import pickle

from langchain_openai import AzureOpenAIEmbeddings, AzureOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.docstore.document import Document
from langchain.vectorstores.faiss import FAISS
import faiss
import streamlit as st

embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
)

faiss_db = "/Users/kkumbier/github/RAG-Chatbot/faiss_db" 
index = FAISS.load_local(
    faiss_db, embedder, allow_dangerous_deserialization=True
)

llm = AzureOpenAI(
    azure_deployment='gpt-35-turbo-instruct'
)

# Generative response
prompt_template = """
    You are a helpful assistant who answers user questions based on multiple contexts given to you.

    Keep your answer short and to the point.
    
    The evidence are the context of the pdf extract with metadata. 
    
    Carefully focus on the metadata 'REF' whenever answering. 
    
    Add citations to the value indicated by REF along with any other relevant citations. 
    
    Format all citations as [<author>, <year>].
        
    Reply "I'm not sure, that information is not in my library" if text is irrelevant.
    
    The PDF content is:
    {pdf_extract}
"""

def make_message(query, prompt_template):
    docs = index.similarity_search(query, k = 5)
    content = "/n ".join([
        d.page_content + " REF: " + d.metadata.get("reference") for d in docs
    ])

    messages = [
        SystemMessage(content = prompt_template.format(pdf_extract = content)), HumanMessage(content = query)
    ]

    return messages

# Initialize streamlit app
st.title(
    ":blue[P]ersister :blue[I]nformation :blue[C]hatbot for :blue[A]I-assisted :blue[R]esearch & :blue[D]evelopment"
)
prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

for message in prompt:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.write(message["content"])

query = st.chat_input("Ask me something about persisters!")

# Add the user's question to the prompt and display it
prompt.append({"role": "user", "content": query})
with st.chat_message("user"):
    st.write(query)

# Display an empty assistant message while waiting for the response
with st.chat_message("picard"):
    botmsg = st.empty()

if query is not None:    
    msg = make_message(query, prompt_template)
    response = []
    result = llm.invoke(msg)
    result.replace('System: ', '')
    response.append(result)
    botmsg.write(result)

    # Add the assistant's response to the prompt
    prompt.append({"role": "picard", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt
    prompt.append({"role": "picard", "content": result})

    # Store the updated prompt in the session state
    st.session_state["prompt"] = prompt

if False:
    qry = "What are cancer persisters?"
    msg = make_message(qry, prompt_template)
    result = llm.invoke(msg)

    qry = "What gene mutations have been studied in cancer persisters?"
    msg = make_message(qry, prompt_template)
    result = llm.invoke(msg)

    qry = "What are some examples of targeted cancer therapies?"
    msg = make_message(qry, prompt_template)
    result = llm.invoke(msg)

    qry = "What is the most common type of cancer?"
    msg = make_message(qry, prompt_template)
    result = llm.invoke(msg)

    qry = "What mechanisms have been proposed to lead to the survival of drug tolerant persister cells?"
    msg = make_message(qry, prompt_template)
    result = llm.invoke(msg)

    qry = "What is the difference between sporadic and familial ALS"
    msg = make_message(qry, prompt_template)
    result = llm.invoke(msg)