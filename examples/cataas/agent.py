import os
import urllib.parse
from typing import Optional, Literal, override, AsyncGenerator
from urllib.parse import urlencode

import dotenv
import instructor
import pydantic
import requests
from a2a.types import AgentAuthentication, AgentCapabilities, AgentCard, AgentSkill
from importlib_metadata import metadata
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field

from ichatbio.agent import IChatBioAgent
from ichatbio.types import Message, TextMessage, ArtifactMessage

dotenv.load_dotenv()


class CataasAgent(IChatBioAgent):
    agent_card: AgentCard

    def __init__(self):
        skill = AgentSkill(
            id="get_random_cat_picture",
            name="Cat picture",
            description="Returns a random cat picture",
            tags=["cat", "picture", "image"],
            examples=["Show me a cat"],
        )

        self.agent_card = AgentCard(
            name="Cat Agent",
            description="Calls the cataas.com API",
            url="http://localhost:9999/",
            version="1.0.0",
            defaultInputModes=["text"],
            defaultOutputModes=["text"],
            capabilities=AgentCapabilities(),
            skills=[skill],
            authentication=AgentAuthentication(schemes=["public"]),
        )

    @override
    async def run(self, request: str, params: dict, **kwargs) -> AsyncGenerator[None, Message]:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        instructor_client = instructor.patch(openai_client)

        try:
            cat = await instructor_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_model=CatModel,
                messages=[{"role": "user", "content": request}],
                max_retries=3
            )

            url = cat.to_url()
            response = requests.get(url)

            yield TextMessage(text="Cat retrieved.")
            yield ArtifactMessage(
                mimetype="image/png",
                description=f"A random cat saying \"{cat.message}\"" if cat.message else "A random cat",
                content=response.content,
                metadata={
                    "api_query_url": url
                }
            )

        except InstructorRetryException as e:
            pass


COLORS = Literal[
    "white", "lightgray", "gray", "black", "red", "orange", "yellow", "green", "blue", "indigo", "violet", "pink"]


class MessageModel(BaseModel):
    """Parameters for adding messages to images."""

    text: str = Field(description="Text to add to the picture.")
    font_size: Optional[int] = Field(None,
                                     description="Font size to use for the added text. Default is 50. 10 is barely readable. 200 might not fit on the picture.")
    font_color: Optional[COLORS] = Field(None, description="Font color to use for the added text. Default is white.",
                                         examples=["red", "green", "yellow", "pink", "gray"])

    @pydantic.field_validator("font_size")
    @classmethod
    def validate_font_size(cls, v):
        if v <= 0:
            raise ValueError("font_size must be positive")
        return v


class CatModel(BaseModel):
    """API parameters for https://cataas.com."""

    tags: Optional[list[str]] = Field(None,
                                      description="One-word tags that describe the cat image to return. Leave blank to get any kind of cat picture.",
                                      examples=[["orange"], ["calico", "sleeping"]])
    message: Optional[MessageModel] = Field(None, description="Text to add to the picture.")

    def to_url(self):
        url = "https://cataas.com/cat"
        if self.tags:
            url += "/" + ",".join(self.tags)
        if self.message:
            url += f"/says/" + urllib.parse.quote(self.message.text)
            params = {}
            if self.message.font_size:
                params |= {"fontSize": self.message.font_size}
            if self.message.font_color:
                params |= {"fontColor": self.message.font_color}
            if params:
                url += "?" + urlencode(params)
        return url
