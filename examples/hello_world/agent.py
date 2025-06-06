from typing import override, Optional, AsyncGenerator

from pydantic import BaseModel

from ichatbio.agent import IChatBioAgent
from ichatbio.types import AgentCard, Message, AgentEntrypoint, TextMessage


class FriendlyAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="The Simplest Agent",
            description="Can only say \"Hello world!\".",
            icon="https://commons.wikimedia.org/wiki/Category:Hello_World#/media/File:Qt_example.png",
            entrypoints=[
                AgentEntrypoint(
                    id="hello",
                    description="Responds with \"Hello world!\".",
                    parameters=None
                )
            ]
        )

    @override
    async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[
        None, Message]:
        yield TextMessage(text="Hello world!")
