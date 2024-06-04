# Persister Information Center for AI-assisted Research and Development
import streamlit as st
import os
import json
import git
import time

from agent.agent_graph import agent
from agent.utils import format_docs
from agent.models import HOLD_LLM

from pandas import read_csv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha


data_dir = os.environ["PERSISTER_DB"]

screens = [
    "012023001-RNASEQ-CELL",
    "042023001-RNASEQ-CELL",
    "122023001-RNASEQ-CELL",
    "052023001-CRISPR-CELL"
]


def generate_thinking(query):
    prompt = PromptTemplate.from_template(template="""
        Generate a statement to let users know you are taking a moment to think about their question. Your statement should be witty and playful
        
        Question: {query}
        """
    )
    
    chain = prompt | HOLD_LLM | StrOutputParser()
    
    return chain.stream({"query": query})

################################################################################
# Initialize streamlit app
################################################################################
score_key = {"yes": 0, "ambiguous": 1, "no":2}
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

picard_avatar = "./avatar.png"

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
def save_response(log_dir, session_id, active_idx, history, score_key):
    
    response_id = int(len(history) / 2)
    score = st.session_state[f"{response_id}_{active_idx}"]
    
    # Update reference score
    refs = history[-1].get("refs")
    refs[active_idx]["score"] = score_key[score]
    history[-1]["refs"] = refs
    
    # Save active chat block to json file
    fout = os.path.join(log_dir, f"log_{session_id}-{response_id}")
    out = {"query": history[-2], "response": history[-1], "githash":sha}
    
    with open(fout, 'w') as f:
        json.dump(out, f, indent=4)


with st.sidebar:
    selected_screen = st.selectbox(
        "Choose a dataset",
        screens
    )

    input_dir = os.path.join(data_dir, selected_screen, "level2")
    
    if "CRISPR" in selected_screen:
        input_file = "crispr.csv"
    else:
        input_file = "gene_de.csv"
    
    
    df_baseline = read_csv(os.path.join(input_dir, input_file))
    st.write(df_baseline)
    
# Print chat history 
for message in history:
    
    if message["content"] is not None:
        
        avatar = picard_avatar if message["role"] == "picard" else None
        
        with st.chat_message(message["role"], avatar = avatar):
            st.write(message["content"])

# Initialize input widget for user queries & append to chat on input
query = st.chat_input("Ask me something about persisters!")

if query is not None:

    # Update history with new query
    history.append({"role": "user", "content": query, "refs": None})
    
    with st.chat_message("user"):
        st.write(query)

    # Invoke agent
    with st.chat_message("picard", avatar=picard_avatar):
        
        st.write_stream(generate_thinking(query))
        result = agent.invoke({"question": query, "df":df_baseline})
        
        if result.get("type") == "generate":
            docs = result.get("documents")
            refs = [{"text": format_docs([d]), "score": 1} for d in docs]
            response = result.get("generation")
        else:
            refs = None
            code = result.get("code")[-1]
            df = df_baseline.copy()
            exec(code)                        
            response = fig
            
        st.write(response)
    
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
                    score_key=score_key
                )
            ) 
            
            # Update reference history in session state
            active_ref["score"] = score_key[choice]
            refs[active_idx] = active_ref
            st.session_state.history[-1]["refs"] = refs
