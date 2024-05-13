# Persister Information Center for AI-assisted Research and Development
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

def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string."""
    formatted = [
        f"Article Reference: {doc.metadata['reference']}\n\n \n\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)

# Initialize embedder used to vectorize queries
embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
)

# Load local vector database and initialize retriever for top-3 docs given query
faiss_db = "/Users/kkumbier/github/RAG-Chatbot/faiss_db" 
index = FAISS.load_local(
    faiss_db, embedder, allow_dangerous_deserialization=True
)

retriever = index.as_retriever(search_kwargs={"k": 3})

# Initialize chatbot LLM
gpt3_5 = AzureChatOpenAI(
    azure_deployment='gpt-35-turbo'
)

def hyde_retriever(query, index, llm, embedder, k=4, q=3):
    
    # Initialize set of related queries
    base_prompt = "Rephrase the following question {question}"
    prompt = ChatPromptTemplate.from_messages(
        [("system", base_prompt), ("user", "{question}")]
    )
    
    chain = prompt | llm | StrOutputParser()
    query_set = set([chain.invoke(query) for i in range(q)])
    
    # Generate pseudo docs for each query
    pa_base_prompt = """
        You are a helpful scientific research assistant who answers user questions. Write a short scientific paper passage to answer the following question. Keep your answer concise and to the point.
        
        QUESTION: {question}
    """

    pa_prompt = ChatPromptTemplate.from_messages(
        [("system", pa_base_prompt), ("user", "{question}")]
    ) 

    chain = pa_prompt | llm | StrOutputParser()
    p_docs = [chain.invoke(q) for q in query_set]
    
    # Embed documents
    d_embeddings = embedder.embed_documents(p_docs)
    q_embeddings = embedder.embed_query(query)
    embeddings = np.array(d_embeddings + [q_embeddings])
    mean_embedding = np.mean(embeddings, axis=0)
    
    # Similarity search for docs
    docs = index.similarity_search_with_score_by_vector(mean_embedding, k=k)
    scores = [d[1] for d in docs]
    docs = [d[0] for d in docs]
    return docs, scores
    
def base_retriever(query, index, k=4):
    out_base = index.similarity_search_with_relevance_scores(query, k=k)
    scores = [d[1] for d in out_base]
    docs = [d[0] for d in out_base]
    return docs, scores

query = "How do cancer stem cells contribute to drug resistance?" 
out_hyde = hyde_retriever(query, index, gpt3_5, embedder)
out_base = base_retriever(query, index)

print(format_docs(out_hyde[0]))
print(format_docs(out_base[0]))


#### SIMPLE RAG ###
gpt4 = AzureChatOpenAI(
    azure_deployment='gpt-4'
)

# Initialize promt for grounded generation
rag_base_prompt = """
    Answer the question below using the provided context. If the context is not relevant to the question, reply "I can't determine that from my available information":

    Context: {context}

    Question: {question}

    Answer:
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [("system", rag_base_prompt), ("user", "{question}")]
)

rag_chain = rag_prompt | gpt4 | StrOutputParser()

answer_hyde = rag_chain.invoke(
    {"context":format_docs(out_hyde[0]), "question":query}
)

answer_base = rag_chain.invoke(
    {"context":format_docs(out_base[0]), "question":query} 
)

# The chain below carries out the following operations:
# - Given a user question, generates a "pseudo response". The psuedo response 
#   is based on the base LLM training data and may contain factual 
#   inaccuracies / hallucinations.
# - Queries a database of scientific research artcile text chunks to select 
#   snippets that are most relevant / similar to the pseudo article.
# - Generates a new response grounded in chunks selected in the previous step.

# Initiallize prompt for pseudoarticle generation
pa_base_prompt = """
    You are a helpful scientific research assistant who answers user questions. Provide an answer to the following question. Keep your answers concise and to the point. Do not include information that is not relevant to the question. Here is the question:
    
    {question}
"""

pa_prompt = ChatPromptTemplate.from_messages(
    [("system", pa_base_prompt), ("user", "{question}")]
)

# Initialize promt for grounded generation
rag_base_prompt = """
    Answer the question below using the provided context. If the context is not relevant to the question, reply "I can't determine that from my available information":

    Context: {context}

    Question: {question}

    Answer:
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [("system", rag_base_prompt), ("user", "{question}")]
)

# Initialize subchain for retrieving article snippets
pa_generator = pa_prompt | llm | StrOutputParser()
pa_retriever = pa_prompt | llm | StrOutputParser() | retriever

# Initialize document formatter
docs_format = itemgetter("docs") | RunnableLambda(format_docs)

# Initialize full chain
answer = rag_prompt | llm | StrOutputParser()

chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=pa_retriever)
    .assign(context=docs_format)
    .assign(answer=answer)
    .pick(["answer", "docs"])
)

# Invoke chain to generate LLM response
query = "How do persisters contribute to drug resistance in cancer?"

retriver_response = retriever.invoke(query)
pa_retriever_response = pa_retriever.invoke(query)

pa = pa_generator.invoke(query)
pa_response = retriever.invoke(f"{query}\n\n{pa}")

print(format_docs(retriver_response))
print(format_docs(pa_retriever_response))
print(pa)
print(format_docs(pa_response))

## Try embedding averager
responses = pa.split("\n\n") + [query]
embedded = embedder.embed_documents(responses)
embedded = np.array(embedded)
embedded = np.mean(embedded, axis = 0)

out = index.similarity_search_by_vector(embedded, k = 3)
print(format_docs(out))


