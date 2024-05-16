from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS

from typing import List
import numpy as np

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string."""
    formatted = [
        f"Article Reference: {doc.metadata['reference']}\n\n \n\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)

class DocRetriver:
  """
  General class for document retrieval.
  """
  
  def __init__(self, **kwargs):
    self.base_retriever = kwargs.get("base_retriever")
    
  def retrieve_docs(self, query, **kwargs):
    docs = self.base_retriever.similarity_search(query, **kwargs)
    
    
    

class HyDocGenerator:
  """ 
  Generates hypothetical documents associated with a query.
  """
  
  def __init__(self, model):
    
    base_prompt = """ 
      You are a helpful scientific research assistant who answers user questions. Write a short paragraph from a scientific paper that answers the following question. Keep your answer concise and to the point.
        
      QUESTION: {question}
    """

    prompt = ChatPromptTemplate.from_messages(
        [("system", base_prompt), ("user", "{question}")]
    ) 
    
    self.chain = chain = prompt | model | StrOutputParser()
    
  def make_hydocs(self, query, h):
    return self.chain.batch([query] * h)
    
class HyDocRetriever:
  """
  Retrieves documents from vector store by matching against hypothetical documents generated from a given query.
  """    

  def __init__(self, **kwargs):
    self.generator = kwargs.get("generator")
    self.index = kwargs.get("index")
    self.embedder = kwargs.get("embedder")
    self.use_query = kwargs.get("use_query")

  def similarity_search(self, query, k=10, h=3):
    
    # Generate hypothetical docs
    hdocs = self.generator.make_hydocs(query, h)
    
    # Embed documents
    embeddings = self.embedder.embed_documents(hdocs)
    
    if self.use_query:
        q_embedding = self.embedder.embed_query(query)
        embeddings = np.array(embeddings + [q_embedding])
    else:
        embeddings = np.array(embeddings)
        
    mean_embedding = np.mean(embeddings, axis=0)
    
    # Similarity search for docs
    return self.index.similarity_search_by_vector(mean_embedding, k=k)

    
def reranker(query, docs, model, d=5):
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

    chain = prompt | model | StrOutputParser()
    
    chat = [{"question":query, "response":d.page_content} for d in docs]
    scores = chain.batch(chat * d)
    scores = np.array(scores).reshape((d, -1))
    return np.mean(scores.astype(int), axis=0)
    

def rerank_selector(query, docs, model, k=3, d=5):
    scores = reranker(query, docs, model, d)
    idcs = np.argpartition(scores, -k)[-k:]
    return [docs[i] for i in idcs]
    
