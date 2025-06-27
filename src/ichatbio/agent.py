from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel

from ichatbio.agent_response import ResponseContext
from ichatbio.types import AgentCard


class IChatBioAgent(ABC):
    """
    An agent capable of using special iChatBio capabilities.
    """

    @abstractmethod
    def get_agent_card(self) -> AgentCard:
        """Returns an iChatBio-specific agent card."""
        pass

    @abstractmethod
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        """
        Receives requests from iChatBio. The `context` object is used to send text responses and initiate data-generating processes.

        :param context: Facilitates response interactions with iChatBio.
        :param request: A natural language description of what the agent should do.
        :param entrypoint: The name of the entrypoint selected to handle this request.
        :param params: Request-related information structured according to the entrypoint's parameter data model.
        :return: A stream of messages.
        """
        pass
