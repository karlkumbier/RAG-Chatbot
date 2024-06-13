SQL_SELECT_TABLE_PROMPT ="""
You are a helpful AI assistant, designed to work with a SQL database. The 
database contains the following tables.

TABLES: {tables}

Determine which of these tables are required to answer the following QUESTION.

QUESTION: {question}

Below is a summary of each table. Use these summaries to determine which tables 
are relevant to the QUESTION.

TABLE SUMMARIES:\n\n{summary}

Return a comman separated list of all TABLES that are relevant to the QUESTION. Your response should only include table names, no additional explanation. If no tables appear to be relevant to the QUESTION, respond `None`.

EXAMPLE RESPONSE: `table-1, table-2`
"""


SQL_QUERY_PROMPT = """
You are an agent designed to interact with a SQL database.

Given an input QUESTION, create a syntactically correct {dialect} query to run.

Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 3 results.

You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for the 
relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

The query must start and end with ```.

The SCHEMA below summarizes tables in the database. Use this schema to formulate your query.

SCHEMA: {db_schema}

QUESTION: {question}
""" 


SQL_DEBUGGER_PROMPT = """
You are an expert software engineer. 

Analyze the following {dialect} QUERY and ERROR MESSAGE(s) to diagnose the
problem with the QUERY. Generate a new, syntactically correct {dialect} query 
that will resolve the error message and answer the original QUESTION.         

The query must start and end with ```.

The SCHEMA below summarizes tables in the database. Use this schema to formulate
your NEW QUERY. 

QUESTION: {question}

ORIGINAL QUERY: {db_query}

SCHEMA: {db_schema}
"""