from typing import override, Optional

import openai
from pydantic import BaseModel

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint, Artifact


class ExamineParameters(BaseModel):
    image: Artifact


class VisionAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Example Vision Agent",
            description="Answers questions about images.",
            icon=None,
            url=None,
            entrypoints=[
                AgentEntrypoint(
                    id="examine",
                    description="Answers questions about a given image.",
                    parameters=ExamineParameters
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        client = openai.Client()

        match params:
            case ExamineParameters(image=image):
                async with context.begin_process(summary="Inspecting image") as process:
                    process: IChatBioAgentProcess

                    await process.log("Finding image URL")
                    # Make sure the artifact is an image
                    if not image.mimetype.startswith("image/"):
                        await process.log("Artifact is not an image")
                        return

                    # Try to find a URL in the image description
                    urls = image.get_urls()
                    if len(urls) == 0:
                        await process.log("Image artifact has no URL")
                        return

                    image_url = urls[0]
                    await process.log(f"Examining image at URL {image_url}")

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
                        await process.log("Analysis: " + response_text)
