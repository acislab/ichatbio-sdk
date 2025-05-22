from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Optional

from ichatbio.types import Message, AgentCard


class IChatBioAgent(ABC):
    @abstractmethod
    def get_agent_card(self) -> AgentCard:
        pass

    @abstractmethod
    async def run(self, request: str, entrypoint: str, params: Optional[dict], **kwargs) -> AsyncGenerator[
        None, Message]:
        """
        :param request: A natural language description of what the agent should do.
        :param entrypoint:
        :param params: Structured data to clarify the request.
        :return: A stream of messages.
        """
        pass
