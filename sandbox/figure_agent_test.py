from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_csv_agent
import pandas as pd

AZURE_OPENAI_DEPLOYMENT_NAME = 'gpt-35-turbo-16k'

x = pd.read_csv("gene_de.csv").filter(
  ["SYMBOL", "CellLine", "pvalue", "log2FoldChange", "Contrast", "ContrastFull"]
)

llm = AzureChatOpenAI(azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME)
agent = create_csv_agent(llm, "gene_de.csv", verbose=True)

agent.metadata

result = agent.run(
  "Generate a plot of -log10(pvalue) versus log2FoldChange for PC9 CellLine"
)
