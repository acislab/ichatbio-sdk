from typing import override, Optional

from pydantic import BaseModel

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.types import AgentCard
from .entrypoints import find_occurrence_records


class IDigBioAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="iDigBio Search",
            description="Searches for information in the iDigBio portal (https://idigbio.org).",
            icon=None,
            url=None,
            entrypoints=[
                # Because this agent is planned to have many entrypoints, we define them in their own files
                find_occurrence_records.entrypoint
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        # Route requests to their selected entrypoints
        match entrypoint:
            case find_occurrence_records.entrypoint.id:
                await find_occurrence_records.run(context, request)
            case _:
                raise NotImplemented(f"{entrypoint} is not yet supported")
