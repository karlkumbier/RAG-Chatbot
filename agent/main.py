from agent_graph import *
from pandas import read_csv

df_baseline = read_csv("../sandbox/gene_de.csv").filter(
    ["SYMBOL", 
    "CellLine", 
    "pvalue", 
    "log2FoldChange", 
    "Contrast", 
    "ContrastFull"
    ]
)

query = "Generate plots of -log10 pvalue versus log2FoldChange for each CellLine. Points should be colored black. The plots should be in one figure with a subplot for each cell line. Points should display the gene SYMBOL when the mouse hovers over them."

#result = plot_chain.invoke({"question":query, "column_names":df_baseline.columns})

result = agent.invoke({
  "question": query,
  "df":df_baseline
})


query = "Generate a scatterplot showing gene SYMBOL on the x-axis and -log10 pvalue on the y-axis. For each cell line, filter the values of SYMBOL to the top 10 genes in terms of -log10 pvalue. The plots should be in one figure with a subplot for each cell line."

result = agent.invoke({
  "question": query,
  "df":df_baseline
})

df = df_baseline.copy()
code = result.get("code")
exec(code[-1])
fig.show()
