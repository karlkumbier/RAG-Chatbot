import json
import langchain.schema
import os
import requests
import sys

from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import AzureChatOpenAI 
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from typing import List

embed_fn = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
)

faiss_db = "/Users/kkumbier/github/RAG-Chatbot/faiss_db" 
db = FAISS.load_local(
    faiss_db, embed_fn, allow_dangerous_deserialization=True
)

llm = AzureChatOpenAI(
    azure_deployment='gpt-35-turbo'
)

# Initialize template for combining chat history
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """
    Given the following conversation, rephrase the follow up question to be a standalone question. 
    Chat History: 
    {chat_history} 
    Follow Up Input: {question} 
    Standalone question:
    """
)

# Initialize template for extracting information from documents
system_template = """
    You are PICARD, a helpful assistant who answers user questions based on multiple context given to you. Start all responses with "PICARD thinks:"
    
    The evidence are the context of the pdf extract with metadata. 
    
    Carefully focus on the metadata reference whenever answering. Add citations based on the reference indicated in context metadata. Format all citations as [<author>, <year>].
        
    Reply "I don't know. That information is not in my library." if text is irrelevant.
    
    The context is:
    {context} 
"""

user_template = "Question: {question}"

messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(user_template)
]

cd_prompt = ChatPromptTemplate.from_messages( messages )

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    output_key='answer', 
    return_messages=True,
)

# Create the ConversationalRetrievalChain with the LLM
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),  
    condense_question_prompt=CONDENSE_QUESTION_PROMPT,
    memory=memory,
    return_source_documents=True, 
    return_generated_question=True, 
    verbose=False, 
    combine_docs_chain_kwargs={"prompt": cd_prompt}
)

def search(user_input):
    query = user_input[-1]['content'] 
    history = [] 

    if len(user_input) == 1: 
        chat_history = "" 
        result = qa({"question": query, 
                     "chat_history": chat_history}
                   )
        response = result["answer"] 
    else:  
        for item in user_input[:-1]: 
            history.append(item["content"]) 
            chat_history = [(history[i], history[i+1]) for i in range(0, len(history), 2)] 
            result = qa({"question": query, 
                         "chat_history": chat_history}
                       )
            response = result["answer"] 
    
    return response

# Function for prompting
def prompt_llm(content):
    user_input = [{"content": content}] 
    response = search(user_input) 
    return response


prompt_llm("What research has been done regarding cancer persisters?")

prompt_llm("What are examples of targeted therapies in cancer?")
prompt_llm("Can you provide citations for these examples?")
prompt_llm("Use the metadata from context documents to indicate which context reference each example came from")