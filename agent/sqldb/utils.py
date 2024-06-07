# Defines functions that are called at SQL agent graph nodes.
from langchain_core.prompts import (
  ChatPromptTemplate, 
  MessagesPlaceholder
)

from langchain.agents.agent_toolkits.base import BaseToolkit

def tool_agent(llm, base_prompt: str, toolkit: BaseToolkit):
  """Create an agent."""
  prompt = ChatPromptTemplate.from_messages([
    ("system", base_prompt),
    MessagesPlaceholder(variable_name="messages"),
  ])
  
  tools = toolkit.get_tools()
  prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
  return prompt | llm.bind_tools(tools)
