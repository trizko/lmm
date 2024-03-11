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
witnesses = [
    Witness.from_json(llm, """{
        "name": "Elizabeth Huntington",
        "role": "Suspect",
        "backstory": "Richard's wife and the matriarch of the family. She was unhappy in her marriage and suspected her husband of having an affair.",
        "alibi": "She claims to have been attending a charity event at the time of the murder.",
        "information": "Elizabeth had recently hired a private investigator to look into her husband's activities."
    }"""),
    Witness.from_json(llm, """{
        "name": "James Huntington",
        "role": "Suspect",
        "backstory": "Richard's eldest son and the heir apparent to the family business. He had a strained relationship with his father due to differing views on running the company.",
        "alibi": "He says he was in a business meeting across town, but his assistant can't confirm his whereabouts during the estimated time of the murder.",
        "information": "James had been arguing with his father about the future direction of the company, and there were rumors of a potential corporate takeover."
    }"""),
    Witness.from_json(llm, """{
        "name": "Sophia Huntington",
        "role": "Suspect",
        "backstory": "Richard's young daughter from his second marriage. She was deeply resentful of her father's favoritism towards her older half-brother James.",
        "alibi": "She claims to have been studying in her room, but the staff saw her wandering the mansion around the time of the murder.",
        "information": "Sophia had recently discovered that her father had changed his will, leaving the bulk of his fortune to James."
    }"""),
    Witness.from_json(llm, """{
        "name": "Benjamin Clark",
        "role": "Murderer",
        "backstory": "The Huntington family's loyal butler who had served them for decades. He was deeply devoted to the late Mrs. Huntington, Richard's first wife.",
        "alibi": "He claims to have been attending to his duties in the kitchen, but there are no witnesses to confirm his whereabouts.",
        "information": "Benjamin resented Richard for remarrying so soon after his beloved first wife's death and believed he had dishonored her memory."
    }"""),
    Witness.from_json(llm, """{
        "name": "Olivia Reynolds",
        "role": "Suspect",
        "backstory": "Richard's personal assistant and rumored mistress. She had been working for him for several years.",
        "alibi": "She says she was running errands for Richard at the time of the murder, but her movements during that period are unaccounted for.",
        "information": "Olivia was recently witnessed having a heated argument with Richard, and there were whispers that he was planning to leave her."
    }""")
]
app = FastAPI()

class HumanInput(BaseModel):
    input: str

@app.post("/chat/")
async def generate_text(human_input: HumanInput):
    return witnesses[0].predict(human_input.input)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
