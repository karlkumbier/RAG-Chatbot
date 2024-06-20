from agent.geordi.agent import GeordiAgent
from agent.rag.agent import RAGAgent
from agent.models import gpt4

from typing import Dict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import (
  ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
)

# TODO: Agents should push plain text description of final output to message 
# thread. E.g., data agent summary should include description of data and 
# plot—this should be impemented in the summarize node of each agent (or 
# potentially in picard summary).
#
# TODO: agent messages may or may not get displayed, depending on desired 
# output. E.g., we may hide the summary message for a figure.
#
# TODO: picard takes chat history (including hidden messages) to generate 
# explicit request, assigns agent to cary out request.
#
# TODO: sub agents should have descriptions of what they do, as configs in assignment prompt
LLM = gpt4

DISTILL_TASK_PROMPT = """
You are the principal investigator of a scientific lab. Based on your 
conversation history, summarize the precise request being made. Your summary 
will be used as a prompt for an AI bot. 

Your summary must clearly and cocisely articulate all tasks that the AI bot needs to perform. 

Your summary should not provide any instructions for how the bot should perform the task.

Vague terms, for example `that` and `it`, should be replaced with the 
explicit thing they are referring to.
"""


prompt = ChatPromptTemplate.from_messages([
  ("system", DISTILL_TASK_PROMPT),
  MessagesPlaceholder(variable_name="messages")
])

chain = prompt | LLM | StrOutputParser()

messages = [
  HumanMessage("What is the forecast temperature in san fancisco next week?"),
  AIMessage("Here are the temperatures: 50, 60, 50, 50, 50, 50, 60"),
  HumanMessage("Can you convert those to celcius?")
]

result = chain.invoke({"messages": messages})

ASSIGN_TASK_PROMPT = """
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