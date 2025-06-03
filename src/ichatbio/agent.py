from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Optional

from pydantic import BaseModel

from ichatbio.types import Message, AgentCard


class IChatBioAgent(ABC):
    """
    Facilitates agent interactions with iChatBio.
    """

    @abstractmethod
    def get_agent_card(self) -> AgentCard:
        """Returns an iChatBio-specific agent card."""
        pass

    @abstractmethod
    async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[None, Message]:
        """
        :param request: A natural language description of what the agent should do.
        :param entrypoint: The name of the entrypoint selected to handle this request.
        :param params: Request-related information structured according to the entrypoint's parameter data model.
        :return: A stream of messages.
        """
        pass
