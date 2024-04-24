# Persister Information Center for AI-assisted Research and Development
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
import streamlit as st

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
        f"Article Reference: {doc.metadata['reference']}\nArticle Snippet: {doc.page_content}"
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
        return result.get("answer") + " [" + refs + "]" 
        

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
prompt_template = """
    You are a helpful assistant who answers user questions based on given context. Reply "I'm not sure, that information is not in my library" if text is irrelevant.
    
    Here is the context:
    {context}
"""

prompt = ChatPromptTemplate.from_messages(
    [("system", prompt_template), ("user", "{question}")]
)

# Initialize subchain for generating an answer once we've done retrieval
answer = prompt | llm | StrOutputParser()

# Initialize complete chain that calls retriver -> formats docs to string -> 
# runs answer subchain -> returns just the answer and retrieved docs.
format = itemgetter("docs") | RunnableLambda(format_docs)

chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format)
    .assign(answer=answer)
    .pick(["answer", "docs"])
)

### Initialize streamlit app ###
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
    response = []
    result = cite_response(chain.invoke(query))
    response.append(result)
    botmsg.write(result)

    # Add the assistant's response to the prompt
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