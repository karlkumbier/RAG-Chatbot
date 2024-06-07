from langchain.agents import create_sql_agent 
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.sql_database import SQLDatabase 
from langchain.llms.openai import OpenAI 
from langchain.agents import AgentExecutor 
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.utilities.sql_database import SQLDatabase
from agent.models import cold_gpt35

# TODO: create user that does not have write permissions
# TODO: reqrite from csv file, may have been overwritten in testing
username = "kkumbier"
password = "persisters"
port = "5432"
host = "localhost"
db = "persisters"

pg_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{db}"
db = SQLDatabase.from_uri(pg_uri)


llm = cold_gpt35
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)


question = "Generate a table of al differential expression results for gene SYMBOL PRMT1 from the 012023001-RNASEQ-CELL screen. Return the SQL query used to generate this table"
