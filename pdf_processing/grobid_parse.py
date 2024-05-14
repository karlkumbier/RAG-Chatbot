import os
import scipdf
import re

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

def docs_from_pdfs(library: str):
  """" Wrapper function for parsing pdf files and generating langchain Documents from a library.

  Args:
      library (str): path to library

  Returns:
      List [Document]: list of langchain documents 
  """
  
  # Initialize papers
  papers = [d for d in os.listdir(library) if not d.startswith(".")]
  
  # Iterate over papers in lab subdirectories and parse docs
  docs = []
  for p in papers:
    pdf_file = os.path.join(library, p)
    docs.append(doc_from_pdf(pdf_file))

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
  chunked_doc = []
  
  title = doc.get("title")
  authors = doc.get("authors")
  year = get_year(doc)
  ref = clean_authors(authors) + ", "+ year

  # rename paper in local library
  file_split = pdf_file.split("/")
  file_split[-1] = f"{ref}.pdf"
  fout = "/".join(file_split).replace(".,", "").replace(" ", "-")
  
  if fout != pdf_file:
    os.rename(pdf_file, fout)
    
  
  # Parse pdf sections
  sections = []
  for s in doc.get("sections"):
    d = Document(page_content=s.get("text"))
    d.metadata["section"] = s.get("heading")
    
    if len(d.page_content):
      sections.append(d)
      

  # Split sections by paragraph
  for s in sections:
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=1,
          separators=["\n\n", "\n"],
          chunk_overlap=0,
      )
      
      chunks = text_splitter.split_text(s.page_content)
      
      for i, chunk in enumerate(chunks):
          
          d = Document(
              page_content=re.sub("^\n*", "", chunk), 
              metadata={"section": d.metadata["section"], "chunk": i}
          )
          
          d.metadata["title"] = title
          d.metadata["reference"] = ref
          d.metadata["authors"] = authors
          d.metadata["type"] = "text"
          chunked_doc.append(d)
  
  # Parse pdf figure captions
  for f in doc.get("figures"):
    label = f.get("figure_label")
    
    if label != "":
      caption = clean_fig_caption(f.get("figure_caption"), label)
      d = Document(page_content=caption)
      d.metadata["figure"] = f"Figure {label}"
      d.metadata["title"] = title
      d.metadata["reference"] = ref
      d.metadata["authors"] = authors
      d.metadata["type"] = "figure"
      chunked_doc.append(d)
  
  return chunked_doc


def clean_fig_caption(caption: str, fig_id: int):
  """ Cleans figure caption by removing Figure X. labels"""
  caption = re.sub(f"^.*Figure {fig_id}.", "", caption) 
  caption = re.sub(f"^.*Fig {fig_id}.", "", caption)
  return re.sub("^ ", "", caption)


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


def get_year(doc: str):
  """ Gets year from paper. If date was not correctly parsed, take latest date of citations as proxy."""
  if doc.get("pub_date") == "" or doc.get("pub_date") is None:
    refs = doc.get("references")
    year = [r.get("year") for r in refs]
    year = [int(y) for y in year if y.isdigit()]
    
    if len(year):
      year = max(year)
    else:
      year = "XXXX"
      
    return(str(year))
  else:
    return doc.get("pub_date").split("-")[0]


if __name__ == "__main__":

  library = "/Users/kkumbier/github/persisters/papers/library"
  db = "/Users/kkumbier/github/RAG-Chatbot/faiss_db"
  docs = docs_from_pdfs(library)

  embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
  )
  
  index = FAISS.from_documents(docs, embedder)
  index.save_local(db)