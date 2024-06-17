RELEVANCE_PROMPT = """
You are a data scientist. Determine whether the table described by TABLE 
SUMMARY contains sufficient information to a answer a user REQUEST. Respond 
with one word: either `sufficient` or `insufficient`. Do not provide any 
explanation. If there is no information in the TABLE SUMMARY below, respond 
`insufficient`.

TABLE_SUMMARY: {df_summary}

REQUEST: {question}
"""

EXPLAIN_IRRELEVANCE_PROMPT = """
You are a data scientist. The following TABLE was deemed insufficient to answer 
a user's REQUEST. Provide an explanation of why the TABLE is insufficient to
address the request, along with more precise instructions to produce a relevant 
table. To help generate your explanation, make use of the TABLE SUMMARY, which provides a natural language description of the table, and SQL QUERY that was used to generate the table.

TABLE: {table}

TABLE SUMMARY: {df_summary}

QUERY: {db_query}
"""

TASK_PROMPT ="""
You are a data scientist tasked with answering questions about your 
laboratories datasets. Given a user REQUEST and message history, determine whether you need to generate a figure or run an analysis. Respond with

`figure` if you need to produce a figure requested by the user

or 

`analysis` if you need to run an analysis requested by the user

Your response should be one word, one of `figure` or `analysis`. Do not provide 
any explanation of your response. 

REQUEST: {question}
"""