# Persister Information Center for AI-assisted Research and Development
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores.faiss import FAISS
import os

# Initialize embedder used to vectorize queries
embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-3-large-1",
)

# Load local vector database and initialize retriever for top-3 docs given query
faiss_db = os.environ["PICARD_DB"] 
index = FAISS.load_local(
    faiss_db, embedder, allow_dangerous_deserialization=True
)

retriever = index.as_retriever(search_kwargs={"k":5})

# Initialize chatbot LLM
gpt4 = AzureChatOpenAI(
    azure_deployment='gpt-4-turbo-128k',
    streaming=True
)

cold_gpt35 = AzureChatOpenAI(
    azure_deployment='gpt-35-turbo-16k',
    temperature=0.5
)

hot_gpt35 = AzureChatOpenAI(
    azure_deployment='gpt-35-turbo-16k',
    temperature=1.25,
    streaming=True
)

RAG_LLM = gpt4
ROUTER_LLM = cold_gpt35
PLOT_LLM = gpt4
DEBUG_LLM = gpt4
ONHOLD_LLM = hot_gpt35
