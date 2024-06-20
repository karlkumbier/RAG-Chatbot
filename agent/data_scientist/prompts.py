RELEVANCE_PROMPT = """
You are a data scientist. Below are sample rows from a TABLE returned from an 
SQL QUERY. Determine whether the TABLE contains sufficient information to a 
answer a user REQUEST. Respond with one word: either `sufficient` or 
`insufficient`. Do not provide any explanation. If there is no information in 
the TABLE below, respond `insufficient`.

To evaluate whehter information in the TABLE is relevant to the REQUEST, 
consider the SQL QUERY used to generate the table in relation to the TABLE 
SCHEMA that describes the TABLE.

TABLE: {table}

TABLE SCHEMA: {schema}

SQL QUERY: {query}

REQUEST: {question}
"""

EXPLAIN_ROUTER_PROMPT = """
You are a data scientist. The following TABLE was deemed insufficient to answer 
a user's REQUEST. Provide an explanation of why the TABLE is insufficient to
address the request, along with more precise instructions to produce a relevant 
table. To help generate your explanation, make use of the SQL QUERY used to generate the table in relation to the TABLE SCHEMA that describes the TABLE.

TABLE: {table}

TABLE SCHEMA: {schema}

QUERY: {query}

REQUEST: {question}
"""

TASK_PROMPT ="""
You are a data scientist tasked with answering questions about your 
laboratories datasets. Given a user REQUEST, determine whether you need to return a figure, table, or run an analysis. Respond with

`figure` if you need to produce a figure requested by the user

`table` if you need to produce a table requested by the user

or 

`analysis` if you need to run an analysis requested by the user

Your response should be one word, one of `figure`, `table`, or `analysis`. Do 
not provide any explanation of your response. 

REQUEST: {question}
"""