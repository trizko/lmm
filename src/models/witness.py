import os
import logging
import json

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

    def __init__(self, llm: ChatOpenAI, name: str, role: str, backstory: str, alibi: str, information: str, is_murderer: bool = False):
        self.name = name
        self.role = role
        self.backstory = backstory
        self.alibi = alibi
        self.information = information
        self.is_murderer = is_murderer
        self.messages = ""

        self.prompt_template = ChatPromptTemplate.from_messages([("system","""
You are an agent in a murder mystery game set in a wealthy family's mansion. The
patriarch of the family, Richard Huntington, has been found dead in his study,
lying face down on the floor with a gunshot wound to the back of his head. The
game involves several agents, one of whom is the murderer...

...The crime scene: Richard Huntington's body was discovered by the butler,
Benjamin Clark, around 9 PM. The study was in disarray, with papers and books
scattered across the floor, suggesting a struggle had taken place. The murder
weapon, a revolver from Huntington's personal collection, was found near the
body. The study door was locked from the inside, with no signs of forced entry.
However, one of the windows was found open, indicating the killer may have
entered or escaped through there.

Forensic evidence: - The gunshot residue pattern on the victim's body and the
position of the wound suggest the shot was fired at close range, from behind. -
There were no other visible injuries on the body besides the gunshot wound. -
Fingerprints were found on the revolver, but they are still being analyzed. - A
broken glass and spilled liquor on the desk suggest Huntington may have been
drinking before the incident.

The rest of the mansion was undisturbed, with the staff and family members
claiming they did not hear any gunshots or commotion from the study during the
estimated time of the murder between 8 PM and 9 PM.

Here are the details about your character:

Your name is {name}
Your role in this story is {role}
Here is your backstory: {backstory}
Here is your alibi for the time of the murder: {alibi}
Information that you know that may be related to the crime: {information}

The following is your conversation with the human player of the game, The
detective:""")])
        self.llm = llm
        self.output_parser = StrOutputParser()
        self.chain = self.prompt_template | self.llm | self.output_parser

    @classmethod
    def from_json(cls, llm: ChatOpenAI, json_data: str):
            """
            Creates a Witness object from a JSON string.

            Args:
                json_data (str): The JSON string representing the witness data.

            Returns:
                Witness: The Witness object created from the JSON data.
            """
            witness_data = json.loads(json_data)
            name = witness_data.get('name')
            role = witness_data.get('role')
            backstory = witness_data.get('backstory')
            alibi = witness_data.get('alibi')
            information = witness_data.get('information')
            return cls(llm, name, role, backstory, alibi, information)

    def invoke(self, input_text):
        """
        Invokes the chain with the given input text and stores the result in the
        history.

        Args:
            input_text (str): The input text to be passed to the chain.

        Returns:
            str: The result obtained from invoking the chain.
        """
        formatted_message = f"Detective: {input_text}\n"
        result = self.chain.invoke({
            "name": self.name,
            "role": self.role,
            "backstory": self.backstory,
            "alibi": self.alibi,
            "information": self.information,
            "messages": self.messages + formatted_message,
        })
        formatted_response = f"You: {result}\n"
        self.messages += formatted_message + formatted_response
        return result
