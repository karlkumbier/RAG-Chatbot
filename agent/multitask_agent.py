from agent.rag_agent import rag_agent
from agent.chart_agent import chart_agent
from agent.sqldb_agent import db_agent
from agent.models import gpt4
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage
from langchain_core.prompts import (
  ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
)

# TODO: database agent responds with nothing relevant when no data tables

# Generate subtask list
PROMPT ="""
You are the principal investigator of a drug discovery laboratory. You 
are tasked with managing a conversation between the following workers.

Data: Data is a data scientist who can generate tables, run analyses, and 
produce figures. Data is familiar with all data sources that are available to 
him. DO NOT direct Data to look up specific datasets or resources. Data will 
decide which datasets or resources are relevant for a given task. DO NOT direct 
Data to run specific analyses. Data will decide which analyses are relevant 
based on the assigned task and available data.


Spock: Spock is an biologist who has expertise in cancer biology. Spock has 
access to a large library of research articles. Spock can report on information 
and results in these articles.

Given a request, break the request down into subtasks that can be answered by 
each worker. Your response should be formatted as a list of key value pairs 
that assign a subtasks workers. The key should indicate the worker and the 
value should indicate a description of the subtask. Your description of subtasks should be as concise as possible.

REQUEST: {question}
"""

prompt = PromptTemplate.from_template(PROMPT)
chain = prompt | gpt4 | StrOutputParser()

question = "Generate a table of gene differential expression for genes that have been implicated in cancer persistence."

tasks = chain.invoke({"question": question})
print(tasks)

# Task assignment
TASK = f"""
You are the principal investigator of a drug discovery laboratory. You have 
assigned the following tasks to a team of workers. 
\n
TASK LIST:\n{tasks}
\n
Workers will perform their assigned task and respond with their results. Given 
the conversation history, determine which tasks have not been completed and 
decide which worker and task to assign next. Respond with one entry 
from the TASK LIST. DO NOT provide any rationale. Just respond with the TASK 
LIST ENTRY. When all tasks have been completed, respond with FINISHED.
"""

prompt = ChatPromptTemplate.from_messages([
  ("system", TASK),
  MessagesPlaceholder(variable_name="messages"),
])

task_chain = prompt | gpt4 | StrOutputParser()
task1 = task_chain.invoke({"messages": []})
print(task1)

gene_list = rag_agent.invoke({"question": task1.split(": ")[1]})
gene_list = gene_list["messages"][-1].content
print(gene_list)

messages = [AIMessage(task1), AIMessage(gene_list)]
task2 = task_chain.invoke({"messages": messages})  