from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader
from azure.ai.documentintelligence.models import DocumentAnalysisFeature
from langchain.text_splitter import MarkdownHeaderTextSplitter

import os

endpoint = os.environ["AZURE_DOCPROC_ENDPOINT"]
key = os.environ["AZURE_DOCPROC_KEY"]
file_path = "/Users/kkumbier/github/persisters/papers/Polyak/Hinohara2018.pdf"

# Load and parse pdf doc
loader = AzureAIDocumentIntelligenceLoader( 
    file_path=file_path, 
    api_endpoint=endpoint, 
    api_key=key, 
    api_model="prebuilt-layout",
    api_version="2024-02-29-preview",
    mode="markdown",
    analysis_features = [DocumentAnalysisFeature.OCR_HIGH_RESOLUTION]
)

parser = loader.parser
doc = loader.load()


# Split the document into chunks base on markdown headers.
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
    ("#####", "Header 5"),
    ("######", "Header 6"),  
    ("#######", "Header 7"), 
    ("########", "Header 8"),
    ("<figure>", "figure")
]

text_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on
)

docs_string = doc[0].page_content
docs_result = text_splitter.split_text(docs_string)

print("Length of splits: " + str(len(docs_result)))