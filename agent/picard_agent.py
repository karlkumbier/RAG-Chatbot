from agent.geordi.agent import GeordiAgent
from agent.rag.agent import RAGAgent
from agent.models import gpt4

from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_core.prompts import (
  ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
)

LLM = gpt4

PICARD_PROMPT = """
You are the principal investigator of a scientific lab. You have two workers that specialize in different areas.

geordi: is a data scientist, who specializes in running analyses on your lab's 
data. Geordi is particularly skilled at generating figures and summary statistics / analyses from your data.

rag: is a biologist who specializes in cancer research. Rag has a deep 
knowledge of biological mechanisms of cancer and is skilled at summarizing 
academic research articles.

You must assign the user REQUEST below to either geordi or rag. Your response 
should be one word, either `geordi` or `rag`, indicating the worker you believe is best suited to carry out the REQUEST.

REQUEST: {question}
"""

prompt = PromptTemplate.from_template(PICARD_PROMPT)
chain = prompt | LLM | StrOutputParser()

chain.invoke("Generate a figure of log fold change versus p-values")
chain.invoke("What are cancer persisters?")

def assign_task(state: Dict, config: Dict) -> Dict:
  llm = config.get("llm", LLM) 
  prompt = PromptTemplate.from_template(PICARD_PROMPT)
  chain = prompt | llm | StrOutputParser()
  return chain.invoke(state)

def run_analysis(state: Dict, config: Dict) -> Dict:
  return None
def