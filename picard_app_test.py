from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS

from operator import itemgetter
from typing import List
import numpy as np

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Initialize embedder used to vectorize queries
embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
)

# Load local vector database and initialize retriever for top-3 docs given query
faiss_db = "/Users/kkumbier/github/RAG-Chatbot/faiss_db_old" 
index = FAISS.load_local(
    faiss_db, embedder, allow_dangerous_deserialization=True
)

# Initialize chatbot LLM
gpt4 = AzureChatOpenAI(
    azure_deployment='gpt-4-turbo-128k'
)

gpt35 = AzureChatOpenAI(
    azure_deployment='gpt-35-turbo-16k'
)

# Initialize helper functions for doc retrieval, scoring, and processing
def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string."""
    formatted = [
        f"Article Reference: {doc.metadata['reference']}\n\n \n\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


def hyde_retriever(query, index, llm, embedder, k=4, q=3, use_query=False):
    """
    Performs document retrieval based on hypothetical documents.
    
    Args:
        query (str): user query
        index: vector store of embedded documents
        llm: large language model used for hypothetical document generation
        k (int): number of documents to be retrieved
        q (int): number of hypothetical documents to generate
        use_query (bool): should query be included with hypothetica documents
            when performing retrieval?   
    """  
    # Generate pseudo docs for each query
    pa_base_prompt = """
        You are a helpful scientific research assistant who answers user questions. Write a short paragraph from a scientific paper that answers the following question. Keep your answer concise and to the point.
        
        QUESTION: {question}
    """

    pa_prompt = ChatPromptTemplate.from_messages(
        [("system", pa_base_prompt), ("user", "{question}")]
    ) 

    chain = pa_prompt | llm | StrOutputParser()
    p_docs = chain.batch([query] * q)
    
    # Embed documents
    d_embeddings = embedder.embed_documents(p_docs)
    
    if use_query:
        q_embeddings = embedder.embed_query(query)
        embeddings = np.array(d_embeddings + [q_embeddings])
    else:
        embeddings = np.array(d_embeddings)
        
    mean_embedding = np.mean(embeddings, axis=0)
    
    # Similarity search for docs
    return index.similarity_search_by_vector(mean_embedding, k=k)

def base_retriever(query, index, k=4):
    return index.similarity_search(query, k=k)

def reranker(query, docs, llm, d=5):
    base_prompt = """
    You are a language model designed to evaluate the responses of this documentation query system.
    
    You will use a rating scale of 0 to 100, 0 being poorest response and 100 being the best.
    
    Responses where the question appears to be answered are considered good.
    
    Responses that include irrelevant information are considered worse than those that are focused on the question.
    
    Responses that provide detailed answers are considered the better than those with limited information on the question.
    
    Also, use your own judgement in analyzing if the question asked is actually answered in the response.
    
    Please rate the question/response pair entered. Only respond with the rating. No explanation necessary. Only integers.
    
    QUESTION: {question}
    
    RESPONSE: {response}
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", base_prompt), 
            ("user", "{question}"), 
            ("ai", "{response}")
        ]
    ) 

    chain = prompt | llm | StrOutputParser()
    
    chat = [{"question":query, "response":d.page_content} for d in docs]
    scores = chain.batch(chat * d)
    scores = np.array(scores).reshape((d, -1))
    return np.mean(scores.astype(int), axis=0)
    

def selector(query, docs, llm, k=3, d=5):
    scores = reranker(query, docs, llm, d)
    idcs = np.argpartition(scores, -k)[-k:]
    return [docs[i] for i in idcs]
    

import time
#query = "How do cancer stem cells contribute to drug resistance?" 
query = "are cancer stem cells a type of persister cell?"

a = time.time()
docs_h35 = hyde_retriever(query, index, gpt35, embedder, q=5, k=10)
b = time.time()
docs_base = base_retriever(query, index, k=10)
c = time.time()

[d.page_content.count("stem") for d in out_hyde_35]
[d.page_content.count("stem") for d in out_base]

docs_h35_f = selector(query, docs_h35, gpt35)
docs_base_f = selector(query, docs_base, gpt35)

print(format_docs(docs_h35_f))
print(format_docs(docs_base_f))


# Initialize promt for grounded generation
rag_base_prompt = """
    Answer the question below using the provided context. 
    
    Do not use information outside of the provided context to answer the question.
    
    If the context is not relevant to the question, reply "I can't determine that from my available information":

    Context: {context}

    Question: {question}

    Answer:
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [("system", rag_base_prompt), ("user", "{question}")]
)

rag_chain = rag_prompt | gpt4 | StrOutputParser()

answer_hyde = rag_chain.invoke(
    {"context":format_docs(docs_h35_f), "question":query}
)

answer_base = rag_chain.invoke(
    {"context":format_docs(docs_base_f), "question":query} 
)
