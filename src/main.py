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
witnesses = [Witness.from_json(llm, """{
    "name": "Elizabeth Huntington",
    "role": "Suspect",
    "backstory": "Richard's wife and the matriarch of the family. She was unhappy in her marriage and suspected her husband of having an affair.",
    "alibi": "She claims to have been attending a charity event at the time of the murder.",
    "information": "Elizabeth had recently hired a private investigator to look into her husband's activities."
}""")]
app = FastAPI()

class HumanInput(BaseModel):
    input: str

@app.post("/chat/")
async def generate_text(human_input: HumanInput):
    return witnesses[0].predict(human_input.input)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
