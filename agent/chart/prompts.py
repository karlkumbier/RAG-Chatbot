MAKE_CHART_CODE_PROMPT = """
You are an expert data scientist. Generate a SINGLE CODE BLOCK that produces a figure to address the QUERY below. Use Pandas and Plotly plots to produce the figure. The dataset is ALREADY loaded into a DataFrame named 'df'. DO NOT load the data again.

The DataFrame has the following columns: {column_names}

Before plotting, ensure that data for any columns used in your solution is ready:

1. Drop any rows with NaN values in any column that will be used in the solution.
2. Check if columns that are supposed to be numeric are recognized as such. If not, attempt to convert them.
3. Perform any transformation of the data using numpy and remove infinite values.


QUERY: {question}

- USE SINGLE CODE BLOCK with a solution. 
- Do NOT EXPLAIN the code 
- DO NOT COMMENT the code. 
- ALWAYS WRAP UP THE CODE IN A SINGLE CODE BLOCK.
- the figure should be stored as a variable named `fig`
- The code block must start and end with ```

"""


DEBUG_CHART_CODE_PROMPT = """
You are an expert software engineer. You have access to a data frame `df` 
with the following columns:

{column_names}

Analyze the following CODE BLOCK and error message(s) in your conversation history to diagnose the problem with the code. Propose a solution that will resolve the error message(s).  Provide a NEW CODE BLOCK that implements the original CODE BLOCK along with the proposed solution. 


- DO NOT explain your proposed solution or the error message(s).
- The NEW CODE BLOCK must start and end with ```
- The NEW CODE BLOCK should seek to perform the same function as the original CODE BLOCK while resolving any errors. 

CODE BLOCK:\n\n{code}
"""


SUMMARIZE_CHART_PROMT = """
You are an expert data scientist. The following CODE BLOCK was used to produce 
a figure from the data frame `df`. Provide a concise, plain text description of the figure based on the CODE BLOCK.

CODE BLOCK: {code}

If the final message incicates an error from the code, your summary should 
state that `No figure could be generated` and describe the error message. DO NOT describe the intended figure if the final message indicates an error. Keep your response under 250 words.

FINAL MESSAGE: {message}
"""

