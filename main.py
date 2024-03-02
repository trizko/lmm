import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

# load environment variables
load_dotenv()

# initialize the llm
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))

# initialize prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])

# initialize output parser
output_parser = StrOutputParser()

# initialize chain
chain = prompt | llm | output_parser

output = chain.invoke({"input": "How do I use the LangChain API?"})
print(output)
