import os
import urllib.parse
from typing import Optional, Literal, override, AsyncGenerator
from urllib.parse import urlencode

import dotenv
import instructor
import pydantic
import requests
from instructor.exceptions import InstructorRetryException
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic import Field

from ichatbio.agent import IChatBioAgent
from ichatbio.types import AgentCard, AgentEntrypoint, ProcessMessage
from ichatbio.types import Message, TextMessage, ArtifactMessage

dotenv.load_dotenv()

CataasResponseFormat = Literal["png", "json"]


class GetCatImageParameters(BaseModel):
    format: CataasResponseFormat = "png"


class CataasAgent(IChatBioAgent):
    def __init__(self):
        self.agent_card = AgentCard(
            name="Cat As A Service",
            description="Retrieves random cat images from cataas.com.",
            icon=None,
            entrypoints=[
                AgentEntrypoint(
                    id="get_cat_image",
                    description="Returns a random cat picture",
                    parameters=GetCatImageParameters
                )
            ]
        )

    @override
    def get_agent_card(self) -> AgentCard:
        return self.agent_card

    @override
    async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[Message, None]:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        instructor_client = instructor.patch(openai_client)

        try:
            yield ProcessMessage(summary="Searching for cats", description="Generating search parameters")

            cat: CatModel = await instructor_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_model=CatModel,
                messages=[
                    {"role": "system",
                     "content": "You translate user requests into Cat-As-A-Service (cataas.com) API parameters."},
                    {"role": "user", "content": request}
                ],
                max_retries=3
            )

            url = cat.to_url(params.format)

            yield ProcessMessage(
                summary="Retrieving cat",
                description=f"Search parameters",
                data={
                    "search_parameters": cat.model_dump(exclude_none=True)
                })

            yield ProcessMessage(description=f"Sending GET request to {url}")

            response = requests.get(url)

            yield ProcessMessage(summary="Cat retrieved", description=f"Received {len(response.content)} bytes")

            yield ArtifactMessage(
                mimetype="image/png",
                description=f"A random cat saying \"{cat.message}\"" if cat.message else "A random cat",
                content=response.content,
                metadata={
                    "api_query_url": url
                }
            )

            yield TextMessage(text="The generated artifact contains the requested image. Note that the artifact's "
                                   "api_query_url returns random images so it should not be considered a location "
                                   "or identifier for the image.")

        except InstructorRetryException as e:
            yield TextMessage(text="Sorry, I couldn't find any cat images.")


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

    def to_url(self, format: CataasResponseFormat):
        url = "https://cataas.com/cat"
        params = {}

        if format == "json":
            params |= {"json": True}

        if self.tags:
            url += "/" + ",".join(self.tags)

        if self.message:
            url += f"/says/" + urllib.parse.quote(self.message.text)
            if self.message.font_size:
                params |= {"fontSize": self.message.font_size}
            if self.message.font_color:
                params |= {"fontColor": self.message.font_color}

        if params:
            url += "?" + urlencode(params)

        return url
