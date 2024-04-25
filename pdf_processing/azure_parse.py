from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

from langchain.docstore.document import Document
import os

endpoint = os.environ["AZURE_DOCPROC_ENDPOINT"]
key = os.environ["AZURE_DOCPROC_KEY"]
file_path="/home/kkumbier/RAG-Chatbot/cabanos2021.pdf"

loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=endpoint, 
    api_key=key, 
    file_path=file_path, 
    api_model="prebuilt-layout"
)

doc = loader.load()