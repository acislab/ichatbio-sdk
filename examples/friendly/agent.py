from datetime import date
from typing import override

from pydantic import BaseModel
from pydantic import PastDate

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard
from ichatbio.types import AgentEntrypoint


class Parameters(BaseModel):
    birthday: PastDate


card = AgentCard(
    name="Friendly Agent",
    description="Responds in a friendly manner.",
    icon="https://example.com/icon.png",
    url=None,
    entrypoints=[
        AgentEntrypoint(
            id="chat",
            description="Generates a friendly reply.",
            parameters=Parameters  # Defined below
        )
    ]
)


class FriendlyAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return card  # The AgentCard we defined earlier

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Parameters):
        if entrypoint != "chat":
            raise ValueError()  # This should never happen

        async with context.begin_process(summary="Replying") as process:
            process: IChatBioAgentProcess

            await process.log("Generating a friendly reply")
            response = ...  # Query an LLM

            await process.log("Response generated", data={"response": response})

            happy_birthday = params.birthday == date.today()
            if happy_birthday:
                await process.log("Generating a birthday surprise")
                audio: bytes = ...  # Generate an audio version of the response
                await process.create_artifact(
                    mimetype="audio/mpeg",
                    description=f"An audio version of the response",
                    content=audio
                )

            await context.reply(
                "I have generated a friendly response to the user's request. For their birthday, I also generated an"
                " audio version of the response."
                if happy_birthday else
                "I have generated a friendly response to the user's request."
            )
