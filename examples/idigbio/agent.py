from typing import override, Optional, AsyncGenerator, AsyncIterator

from pydantic import BaseModel

from ichatbio.agent import IChatBioAgent
from ichatbio.types import AgentCard, Message
from .entrypoints import find_occurrence_records


class IDigBioAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="iDigBio Search",
            description="Searches for information in the iDigBio portal (https://idigbio.org).",
            icon=None,
            entrypoints=[
                # Because this agent is planned to have many entrypoints, we define them in their own files
                find_occurrence_records.entrypoint
            ]
        )

    @override
    async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[None, Message]:
        # Route requests to their selected entrypoints
        coro: AsyncIterator[Message]
        match entrypoint:
            case find_occurrence_records.entrypoint.id:
                coro = find_occurrence_records.run(request)
            case _:
                raise ValueError()

        # Execute the entrypoint code (an asyncio coroutine)
        async for message in coro:
            yield message
