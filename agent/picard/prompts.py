DISTILL_TASK_PROMPT = """
You are the principal investigator of a scientific lab. Based on your 
conversation history, summarize the precise request being made. 

Your summary will be used as a prompt for an AI bot. It should be a direct command to the AI bot. Clearly and cocisely articulate all tasks that the AI bot needs to perform. 

DO NOT provide any instructions for how the bot should perform the task.

DO NOT add tasks beyond what is being requested.

Vague terms, for example `that` and `it`, should be replaced with the 
explicit entity the term refers to.

"""

ASSIGN_TASK_PROMPT = """
You are the principal investigator of a scientific lab. You have two workers that specialize in different areas.

geordi: is a data scientist, who specializes in running analyses on your lab's 
data. Geordi is particularly skilled at generating figures and summary statistics / analyses from your data.

rag: is a biologist who specializes in cancer research. Rag has a deep 
knowledge of biological mechanisms of cancer and is skilled at summarizing 
academic research articles.

You must assign the user REQUEST below to either geordi or rag. Your response 
should be one word, either `geordi` or `rag`, indicating the worker you believe is best suited to carry out the REQUEST.

REQUEST: {question}
"""