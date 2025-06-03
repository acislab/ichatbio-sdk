from datetime import date
from typing import Optional, override, AsyncGenerator

from pydantic import BaseModel
from pydantic import PastDate

from ichatbio.agent import IChatBioAgent
from ichatbio.types import AgentCard, Message, TextMessage, ProcessMessage, ArtifactMessage
from ichatbio.types import AgentEntrypoint


class ChatParameters(BaseModel):
    birthday: PastDate


card = AgentCard(
    name="Friendly Agent",
    description="Responds in a friendly manner.",
    icon="https://example.com/icon.png",
    entrypoints=[
        AgentEntrypoint(
            id="chat",
            description="Generates a friendly reply.",
            parameters=ChatParameters  # Defined below
        )
    ]
)


class FriendlyAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return card  # The AgentCard we defined earlier

    @override
    async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[
        None, Message]:
        if entrypoint != "chat":
            raise ValueError()  # This should never happen

        yield ProcessMessage(summary="Replying",
                             description="Generating a friendly reply")
        response = ...  # Query an LLM

        yield ProcessMessage(description="Response generated",
                             data={"response": response})

        happy_birthday = ChatParameters(params).birthday == date.today()
        if happy_birthday:
            yield ProcessMessage(description="Generating a birthday surprise")
            audio: bytes = ...  # Generate an audio version of the response
            yield ArtifactMessage(
                mimetype="audio/mpeg",
                description=f"An audio version of the response",
                content=audio)

        yield TextMessage(
            "I have generated a friendly response to the user's request. For their birthday, I also generated an audio version of the response."
            if happy_birthday else
            "I have generated a friendly response to the user's request.")
