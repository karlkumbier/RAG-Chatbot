# Persister Information Center for AI-assisted Research and Development
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
import os
import time
import json
import cohere
import git

from rag import RAG, format_docs

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

# Initialize chatbot LLM
gpt4 = AzureChatOpenAI(
    azure_deployment='gpt-4-turbo-128k',
    streaming=True
)

gpt35 = AzureChatOpenAI(
    azure_deployment='gpt-35-turbo-16k',
    temperature=1.25,
    streaming=True
)

co = cohere.Client(os.environ["COHERE_API_KEY"])


ragbot = RAG(
    model=gpt4,
    index=index,
    hyde_model=gpt35,
    hyde_embedder=embedder,
    rerank_model=co
)

def generate_response(query, context, chain):
    return chain.stream({"question": query, "context":context})

def generate_thinking(query):
    base_prompt = """
        Generate a statement to let users know you are taking a moment to think about their question. Your statement should be witty and playful
        
        Question: {query}
        """
    # Test that streaming works using non AzureOpenAI
    #for event in co.chat_stream(message=base_prompt):
    #    if event.event_type == "text-generation":
    #        yield(event.text)
    prompt = ChatPromptTemplate.from_messages(
        [("system", base_prompt), ("user", "{query}")])
    chain = prompt | gpt35 | StrOutputParser()
    
    return chain.stream({"query": query})

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
    
    response_id = int(len(history) / 2)
    score = st.session_state[f"{response_id}_{active_idx}"]
    
    # Update reference score
    refs = history[-1].get("refs")
    refs[active_idx]["score"] = skey[score]
    history[-1]["refs"] = refs
    
    # Save active chat block to json file
    fout = os.path.join(log_dir, f"log_{session_id}-{response_id}")
    out = {"query": history[-2], "response": history[-1], "githash":sha}
    
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
    docs = ragbot.get_context(query, k=5)
    context = format_docs(docs)

    with st.chat_message("picard", avatar="🖖"):
        
        st.write_stream(generate_thinking(query))
        
        response = st.write_stream(
            generate_response(query, context, ragbot.rag_chain)
        )

    refs = [{"text": format_docs([doc]), "score": 1} for doc in docs] 
    
    # Update history with picard's response
    history.append({
        "role": "picard", 
        "content": response, 
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
                key=f"{int(len(st.session_state.history) / 2)}_{active_idx}",
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