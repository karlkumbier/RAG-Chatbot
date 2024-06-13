CHART_PROMPT = """
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


DEBUG_CHART_PROMPT = """
    You are an expert software engineer. Analyze the following CODE BLOCK and
    ERROR MESSAGE to diagnose the problem with the code. Propose a solution that
    will resolve the error message.  Provide NEW CODE BLOCK that implements the original CODE block along with the proposed solution. Provide any explanation for your prosed solution after the NEW CODE BLOCK.
    
    - The NEW CODE BLOCK must start and end with ```
    - The NEW CODE BLOCK should seek to perform the same function as the original CODE BLOCK while resolving any errors. 
"""