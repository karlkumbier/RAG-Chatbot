# Persister Information Center for AI-assisted Research and Development
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS

from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string."""
    formatted = [
        f"TITLE: {doc.metadata['title']}\n AUTHORS: {doc.metadata['authors']}\n REFERENCE: {doc.metadata['reference']}\n Article content: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted)

def cite_response(result: str) -> str:
    """Reference answer based on doc(s) used to generate response"""
    refs = [d.metadata.get("reference") for d in result.get("docs")]
    refs = "; ".join(set(refs))

    if result.get("answer") == "I'm not sure, that information is not in my library":
        return result.get("answer")        
    else:
        return result.get("answer") + " [" + refs + "]" 
        

# Initialize embedder used to vectorize queries
embedder = AzureOpenAIEmbeddings(
    deployment="text-embedding-ada-002",
)

# Load local vector database and initialize retriever for top-3 docs given query
faiss_db = "/Users/kkumbier/github/RAG-Chatbot/faiss_db" 
index = FAISS.load_local(
    faiss_db, embedder, allow_dangerous_deserialization=True
)

retriever = index.as_retriever(k = 3)

# Initialize chatbot LLM
llm = AzureChatOpenAI(
    azure_deployment='gpt-35-turbo'
)

# Generative response
base_prompt = """
    You are a helpful assistant who answers user questions based on given context from selected articles. Reply "I'm not sure" if the context given to you is irrelevant to the question. 
    
    You must return both an answer and citation(s). A citation consists of a VERBATIM quote from the article that justifies the answer and the REFERENCE indicated at the beginning of the quoted article. Return a citation for every quote across all articles that justify the answer.
    
    Format your response as follows:
    
    ANSWER [REFERENCE]
    
    VERBATIM QUOTE(S) FROM REFERENCE [REFERENCE]
    
    Here is the context:
    {context}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", base_prompt), ("user", "{question}")]
)

# Initialize subchain for generating an answer once we've done retrieval
answer = prompt_template | llm | StrOutputParser()

# Initialize complete chain that calls retriver -> formats docs to string -> 
# runs answer subchain -> returns just the answer and retrieved docs.
format = itemgetter("docs") | RunnableLambda(format_docs)

chain = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format)
    .assign(answer=answer)
    .pick(["answer", "docs"])
)

example_q = "What is a cancer persister"
result = chain.invoke(example_q)

################################################################################
# Citations through function calling
################################################################################
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

class cited_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )

def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nReference: {doc.metadata['reference']}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)

llm_with_tool = llm.bind_tools(
    [cited_answer],
    tool_choice="cited_answer",
)

output_parser = JsonOutputKeyToolsParser(
    key_name="cited_answer", 
    first_tool_only=True
)

format_1 = itemgetter("docs") | RunnableLambda(format_docs_with_id)
answer_1 = prompt_template | llm_with_tool | output_parser

chain_1 = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format_1)
    .assign(cited_answer=answer_1)
    .pick(["cited_answer", "docs"])
)

result = chain_1.invoke(example_q)

################################################################################
# Citing specific blocks of text
################################################################################
class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class quoted_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )    
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )
    
    
output_parser_2 = JsonOutputKeyToolsParser(
    key_name="quoted_answer", first_tool_only=True
)

llm_with_tool_2 = llm.bind_tools(
    [quoted_answer],
    tool_choice="quoted_answer",
)

format_2 = itemgetter("docs") | RunnableLambda(format_docs_with_id)
answer_2 = prompt_template | llm_with_tool_2 | output_parser_2

chain_2 = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format_2)
    .assign(quoted_answer=answer_2)
    .pick(["quoted_answer", "docs"])
)

# Parse result
result = chain_2.invoke(example_q)
answer = result.get("quoted_answer").get("answer")
refs = [d.metadata.get("reference") for d in result.get("docs")]

citations = {
    "[{}] {}".format(r.get("source_id"), refs[r.get("source_id")]):r.get("quote")
    for r in result.get("quoted_answer").get("citations")
}

################################################################################
# Citations through prompting
################################################################################
from langchain_core.output_parsers import XMLOutputParser

base_prompt = """
You're a helpful AI assistant. Given a user question and some context, answer 
the user question and provide citations. If none of the articles answer the 
question, just say you don't know.

You must return both an answer and citations. A citation consists of a VERBATIM 
quote that justifies the answer and the ID of the quote article. Return a 
citation for every quote across all articles that justify the answer. Use the 
following format for your final output:

<cited_answer>
    <answer></answer>
    <citations>
        <citation><source_id></source_id><quote></quote></citation>
        <citation><source_id></source_id><quote></quote></citation>
        ...
    </citations>
</cited_answer>

Here is the context:
{context}
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", base_prompt), ("user", "{question}")]
)

def format_docs_xml(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        doc_str = f"""\
    <source id=\"{i}\">
        <title>{doc.metadata['reference']}</title>
        <article_snippet>{doc.page_content}</article_snippet>
    </source>"""
        formatted.append(doc_str)
    return "\n\n<sources>" + "\n".join(formatted) + "</sources>"


format_3 = itemgetter("docs") | RunnableLambda(format_docs_xml)
answer_3 = prompt_template | llm

chain_3 = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever)
    .assign(context=format_3)
    .assign(cited_answer=answer_3)
    .pick(["cited_answer", "docs"])
)

result = chain_3.invoke(example_q)