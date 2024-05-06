# Persister Information Center for AI-assisted Research and Development
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS

import streamlit as st
import os
import time
import json

from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string."""
    formatted = [
        f"Article Reference: {doc.metadata['reference']}\n\n \n\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)

def cite_response(result: str) -> str:
    """Reference answer based on doc(s) used to generate response"""
    refs = [d.metadata.get("reference") for d in result.get("docs")]
    refs = "; ".join(set(refs))

    if result.get("answer") == "I'm not sure, that information is not in my library":
        return result.get("answer")        
    else:
        return f"{result.get("answer")} [{refs}]", result.get("docs") 
        

# Initialize embedder used to vectorize queries
embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
)

# Load local vector database and initialize retriever for top-3 docs given query
faiss_db = "/Users/kkumbier/github/RAG-Chatbot/faiss_db" 
index = FAISS.load_local(
    faiss_db, embedder, allow_dangerous_deserialization=True
)

retriever = index.as_retriever(k = 3)

# Initialize chatbot LLM
llm = AzureChatOpenAI(
    azure_deployment='gpt-35-turbo'
)

# Generative response
base_prompt = """
    You are a helpful assistant who answers user questions based on given context. Reply "I'm not sure, that information is not in my library" if text is irrelevant.
    
    Here is the context:
    {context}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", base_prompt), ("user", "{question}")]
)

# Initialize subchain for generating an answer once we've done retrieval
answer = prompt_template | llm | StrOutputParser()

# Initialize complete chain that calls retriver -> formats docs to string -> 
# runs answer subchain -> returns just the answer and retrieved docs.
format = itemgetter("docs") | RunnableLambda(format_docs)

chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format)
    .assign(answer=answer)
    .pick(["answer", "docs"])
)

################################################################################
# Initialize streamlit app
################################################################################
skey = {"yes": 0, "ambiguous": 1, "no":2}
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

st.title(
    ":blue[P]ersister :blue[I]nformation :blue[C]hatbot for :blue[A]I-assisted :blue[R]esearch & :blue[D]evelopment"
)

# Initialize list for chat and references
history = st.session_state.get("history", [])
refs = st.session_state.get("refs", None)

# Initialize session_id for logging responses
session_id = st.session_state.get("id", time.time())
st.session_state.id = session_id

# Initialize reference index and callbacks
if "ref_idx" not in st.session_state:
    st.session_state.ref_idx = 0

def increase_ref_idx():
    st.session_state.ref_idx += 1
    
def decrease_ref_idx():
    st.session_state.ref_idx -= 1

# Initialize reference toggle button and callbacks
if 'show_refs' not in st.session_state:
    st.session_state.show_refs = False
    
def click_show_refs():
    st.session_state.show_refs = not st.session_state.show_refs

# Initialize response scoring callbacks
def save_response(log_dir, session_id, active_idx, history, skey):
    
    response_id = len(history)    
    score = st.session_state[f"{response_id}_{active_idx}"]
    
    # Update reference score
    refs = history[-1].get("refs")
    refs[active_idx]["score"] = skey[score]
    history[-1]["refs"] = refs
    
    # Save active chat block to json file
    fout = os.path.join(log_dir, f"log_{session_id}-{response_id}")
    out = {"query": history[-2], "response": history[-1]}
    
    with open(fout, 'w') as f:
        json.dump(out, f, indent=4)

# Print chat history 
for message in history:
    if message["content"] is not None:
        avatar = "🖖" if message["role"] == "picard" else None
        with st.chat_message(message["role"], avatar = avatar):
            st.write(message["content"])

# Initialize input widget for user queries & append to chat on input
query = st.chat_input("Ask me something about persisters!")

if query is not None:

    # Update history with new query
    history.append({"role": "user", "content": query, "refs": None})
    
    with st.chat_message("user"):
        st.write(query)

    # Invoke chain to generate LLM response
    answer, refs = cite_response(chain.invoke(query))
    refs = [{"text": format_docs([doc]), "score": 1} for doc in refs] 
    
    with st.chat_message("picard", avatar="🖖"):
        st.write(answer)

    # Update history with picard's response
    history.append({
        "role": "picard", 
        "content": answer, 
        "refs": refs
    })
    
    # Update session state    
    st.session_state.history = history
    st.session_state.refs = refs
    st.session_state.ref_idx = 0
    st.session_state.show_refs = False
    


if refs is not None:
    
    st.button('Toggle references', on_click=click_show_refs)
    
    if st.session_state.show_refs:

        with st.container(height=300):
            active_idx = st.session_state.ref_idx % len(refs) 
            active_ref = refs[active_idx]
            st.write(active_ref.get("text"))
            
            cols = st.columns([1, 1, 1, 1, 1])
            
            with cols[0]:
                st.button(
                    "Previous", 
                    use_container_width=True, 
                    on_click=decrease_ref_idx
                )
            with cols[-1]:
                st.button(
                    "Next", 
                    use_container_width=True,
                    on_click=increase_ref_idx
                )
            
            rscore = st.empty()
            choice = rscore.radio(
                "Is this reference relevant to the response?", 
                ["yes", "ambiguous", "no"], 
                index=active_ref.get("score"),
                key=f"{len(st.session_state.history)}_{active_idx}",
                on_change=save_response,
                kwargs=dict(
                    log_dir=log_dir, 
                    session_id=session_id, 
                    active_idx=active_idx,
                    history=history,
                    skey=skey
                )
            ) 
            
            # Update reference history in session state
            active_ref["score"] = skey[choice]
            refs[active_idx] = active_ref
            st.session_state.history[-1]["refs"] = refs