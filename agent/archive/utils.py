import re
from langchain_core.documents import Document
from typing import List

def extract_python_code(text: str) -> str:
  """Extract formatted block of python code callable by `exec`"""  
  text = text.replace("```python", "```")
  pattern = r'```\s(.*?)```'
  matches = re.findall(pattern, text, re.DOTALL)
  if not matches:
    return None
  else:
    return matches[0]


def format_docs(docs: List[Document]) -> str:
  """Convert list of Documents to a formated string for LLM ingestion"""
  formatted = [
    f"Article Reference: {doc.metadata['reference']}\n\n \n\nArticle Snippet: {doc.page_content}"
    for doc in docs
  ]
  return "\n\n" + "\n\n".join(formatted)
