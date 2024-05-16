from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from typing import List
import numpy as np
import cohere
import os

# Initialize embedder used to vectorize queries
embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-large-1",
)

# Load local vector database and initialize retriever for top-3 docs given query
faiss_db = "/Users/kkumbier/github/RAG-Chatbot/faiss_db" 
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

co = cohere.Client(os.environ["COHERE_API_KEY"])

class RAG:
    """ Implements retrieval augmented generation system. """
    def __init__(self, **kwargs):
        
        # Initialize required parameters
        if kwargs.get("index") is None:
            raise Exception("specify index for retrieval")
        else:
            self.index = kwargs.get("index")
            

        # Initialize optional parameters
        self.hyde_model = kwargs.get("hyde_model")
        self.hyde_embedder = kwargs.get("hyde_embedder")
        self.hyde_q = kwargs.get("hyde_q")
        self.rerank_model = kwargs.get("rerank_model")

        # Initialize RAG chain
        if kwargs.get("model") is None:
            raise Exception("specify model for generation")
        
        rag_base_prompt = """
            Answer the question below using the provided context. The context provided are snippets from academic articles, with ARTICLE REFERENCE, formatted as AUTHOR (YEAR), indicated before each snippet. Cite any part of your answer that depends on a provided snippet using the ARTICLE REFERENCE associated with each snippet. Use inline citations formatted as (AUTHOR, YEAR)
            
            Do not use information outside of the provided context to answer the question.
            
            If the context is not relevant to the question, reply "I can't determine that from my available information":

            Context: {context}

            Question: {question}

            Answer:
        """

        rag_prompt = ChatPromptTemplate.from_messages(
            [("system", rag_base_prompt), ("user", "{question}")]
        )

        self.rag_chain = rag_prompt | kwargs.get("model") | StrOutputParser()

        
    def retrieve(self, query, k, hyde=False):
        """Performs document retrieval making a call to hyde/base retriever"""
        
        if hyde:
            
            if self.hyde_model is None or self.hyde_embedder is None:
                raise Exception
            
            if self.hyde_q is None:
                self.hyde_q = 3
                    
            docs = hyde_retrieve(
                query, 
                self.index, 
                self.hyde_model, 
                self.hyde_embedder, 
                k,
                self.hyde_q
            )
        
        else:
            docs = base_retrieve(
                query,
                self.index,
                k
            )
            
        return docs


    def generate_response(self, query, k, hyde=False, rerank=False, rerank_k=5):
        """ Makes calls to retriever, reranker (optionally) and RAG chain to generate responses to user queries."""
        
        docs = self.retrieve(query, k, hyde)
        
        # Filter docs based on cohere reanker if specified
        if rerank:
            if self.rerank_model is None:
                raise Exception            
            docs = select_reranked(query, docs, self.rerank_model, rerank_k)

        response = self.rag_chain.invoke(
            {"context":format_docs(docs), "question":query}
        )
        
        return response, docs   

def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string."""
    formatted = [
        f"Article Reference: {doc.metadata['reference']}\n\n \n\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)


def hyde_retrieve(query, index, model, embedder, k=4, q=3, use_query=False):
    """
    Performs document retrieval based on hypothetical documents.
    
    Args:
        query (str): user query
        index: vector store of embedded documents
        model: large language model used for hypothetical document generation
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

    chain = pa_prompt | model | StrOutputParser()
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

def base_retrieve(query, index, k=4):
    return index.similarity_search(query, k=k)


def select_reranked(query, docs, model, k=5):
    docstr = [d.page_content for d in docs]

    response = model.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=docstr,
        top_n=k,
    )

    return [docs[r.index] for r in response.results]


#query = "How do cancer stem cells contribute to drug resistance?" 
#query = "Are cancer stem cells a type of persister cell?"
#query = "what is the relationship between IGF1R and cancer persisters?"
#query = "what is the role of the YAP/TEAD pathway in cancer persistence?"
#query = "what is the most common genetic mutation in ALS patients?"
#query = "list the most common mutations in NSCLC patients"