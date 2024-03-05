import json
import logging
import os

import uvicorn

from dotenv import load_dotenv
from fastapi import FastAPI, Body
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from models.witness import Witness


load_dotenv()
logger = logging.getLogger("uvicorn")


llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"))
witness = Witness(llm)
app = FastAPI()

class InputText(BaseModel):
    input_text: str

@app.post("/chat/")
async def generate_text(input_text: InputText):
    return witness.invoke({"input": input_text})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
