# Persister Information Center for AI-assisted Research and Development
from langchain_openai import AzureChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent

import git

repo = git.Repo(search_parent_directories=True)
sha = repo.head.object.hexsha

# Initialize chatbot LLM
gpt4 = AzureChatOpenAI(
    azure_deployment='gpt-4-turbo-128k'
)

input_file = "gene_de.csv"

agent = create_csv_agent(
    gpt4,
    input_file,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

result = agent.run("What are the values of 'ContrastFull'?")
result = agent.run("""
  Generate a table of gene 'SYMBOL' where 'log2FoldChange' > 1 and 'padj' < 0.05 for the T1_vs_T0 'Contrast'"""
)