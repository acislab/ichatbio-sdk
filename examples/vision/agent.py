from typing import override, Optional, AsyncGenerator

import openai
from pydantic import BaseModel

from ichatbio.agent import IChatBioAgent
from ichatbio.types import AgentCard, Message, AgentEntrypoint, Artifact, ProcessMessage


class ExamineParameters(BaseModel):
    image: Artifact


class VisionAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Example Vision Agent",
            description="Answers questions about images.",
            icon=None,
            url="http://localhost:9999",
            entrypoints=[
                AgentEntrypoint(
                    id="examine",
                    description="Answers questions about a given image.",
                    parameters=ExamineParameters
                )
            ]
        )

    @override
    async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[
        None, Message]:
        client = openai.Client()

        match params:
            case ExamineParameters(image=image):
                yield ProcessMessage(summary="Examining image", description="Finding image URL")
                # Make sure the artifact is an image
                if not image.mimetype.startswith("image/"):
                    yield ProcessMessage(description="Artifact is not an image")
                    return

                # Try to find a URL in the image description
                urls = image.get_urls()
                if len(urls) == 0:
                    yield ProcessMessage(description="Image artifact has no URL")
                    return

                image_url = urls[0]
                yield ProcessMessage(description=f"Examining image at URL {image_url}")

                # Ask GPT-4o-mini to answer the request
                response = client.responses.create(
                    model="gpt-4o-mini",
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": request},
                                {"type": "input_image", "image_url": image_url}
                            ]
                        }
                    ]
                )

                response_text = response.output_text
                if response_text:
                    yield ProcessMessage(description="Analysis: " + response_text)
