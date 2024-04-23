import os
import scipdf
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

def docs_from_pdfs(library: str):
  """" Wrapper function for parsing pdf files and generating langchain Documents from a library. Library is assumed to be organized as a set of subject-specific subdirectories (e.g., labs, topics, etc.), each subdirectory containing pdf files.

  Args:
      library (str): path to library

  Returns:
      List [Document]: list of langchain documents 
  """
  
  # Initialize lab directories
  lab_dirs = [d for d in os.listdir(library) if not d.startswith(".")]
  
  # Iterate over papers in lab subdirectories and parse docs
  docs = []
  for d in lab_dirs:
    papers = os.listdir(os.path.join(library, d))
    for p in papers:
      docs = docs + doc_from_pdf(os.path.join(library, d, p))
      
  return docs
    
def doc_from_pdf(pdf_file: str):
  """ Parses pdf files first by section, then chunks sections into maximum 4000 
  character Documents. Metadata for documents includes: paper title, authors, year, section, chunk.

  Args:
      pdf_file (str): path to pdf file

  Returns:
      List [Document]: list of langchain documents
  """
  # Parse pdf doc
  doc = scipdf.parse_pdf_to_dict(pdf_file) 
  title = doc.get("title")
  authors = doc.get("authors")
  year = get_year(doc.get("pub_date"))
  ref = clean_authors(authors) + ", "+ year

  # TODO: abstract processing 
  # Parse pdf by section
  sec_docs = []
  for s in doc.get("sections"):
    d = Document(page_content = s.get("text"))
    d.metadata["section"] = s.get("heading")
    
    if len(d.page_content):
      sec_docs.append(d)
    
  # Split sections by paragraph for maximal sized doc
  doc_chunks = []
  for d in sec_docs:
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=4000,
          separators=["\n"],
          chunk_overlap=0,
      )
      
      chunks = text_splitter.split_text(d.page_content)
      for i, chunk in enumerate(chunks):
          d = Document(
              page_content=chunk, 
              metadata={"section": d.metadata["section"], "chunk": i}
          )
          
          d.metadata["title"] = title
          d.metadata["reference"] = ref
          d.metadata["authors"] = authors
          doc_chunks.append(d)

  return doc_chunks


def clean_authors(author_list: str):
  """ Cleans list of authors for standard referencing format"""
  author_list = author_list.split("; ")
  author_list = [a.split(" ")[-1] for a in author_list]
  
  if len(author_list) == 1:
    return author_list[0]
  elif len(author_list) == 2:
    return " and ".join(author_list)
  else:
    return author_list[0] + " et al."
  

def get_year(date: str):
  """ Gets year from date string"""
  return date.split("-")[0]


if __name__ == "__main__":
  
  library = "/Users/kkumbier/github/persisters/papers/"
  db = "/Users/kkumbier/RAG-Chatbot/faiss_db"
  docs = docs_from_pdfs(library)

  embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
  )
  
  index = FAISS.from_documents(docs, embedder)
  index.save_local(db)