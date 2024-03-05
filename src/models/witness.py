import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
logger = logging.getLogger("uvicorn")

class Witness:
    """
    The Witness model is used to store information about a witness to a crime.
    It is essentially an LLM chain with a specific prompt and output parser.
    It also contains the history of the conversation with the main player.
    """

    def __init__(self, llm: ChatOpenAI):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a witness to a crime."),
            ("user", "{input}")
        ])
        self.llm = llm
        self.output_parser = StrOutputParser()
        self.chain = self.prompt | self.llm | self.output_parser
        self.history = []

    def invoke(self, input_text):
        """
        Invokes the chain with the given input text and stores the result in the
        history.
        """
        result = self.chain.invoke({"input": input_text})
        self.history.append((input_text, result))
        return result

    def get_history(self):
        """
        Returns the conversation history.
        """
        return self.history