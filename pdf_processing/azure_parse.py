from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

import os

endpoint = "https://awlab-docparse.cognitiveservices.azure.com/"
#os.environ["AZURE_DOCPROC_ENDPOINT"]
key = "f7f6c8a9405648928a6a1240352f2e32" 
#os.environ["AZURE_DOCPROC_KEY"]
file_path="/Users/karlkumbier/github/RAG-Chatbot/cabanos2021.pdf"

loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=endpoint, 
    api_key=key, 
    file_path=file_path, 
    api_model="prebuilt-layout"
)

doc = loader.load()
