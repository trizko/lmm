import json
import logging
import os

import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
logger = logging.getLogger("uvicorn")


# initialize the chain of langchain components
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "{input}")
])
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
output_parser = StrOutputParser()
chain = prompt | llm | output_parser


app = FastAPI()


@app.get("/chat/")
async def generate_text():
    return chain.invoke({"input": "I need to write a technical documentation for a new software."})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
