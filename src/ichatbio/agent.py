from abc import abstractmethod
from collections.abc import AsyncGenerator

from ichatbio.types import Message


class IChatBioAgent:
    @abstractmethod
    async def run(self, request: str, params: dict, **kwargs) -> AsyncGenerator[None, Message]:
        """
        :param request: A natural language description of what the agent should do.
        :param params: Structured data to clarify the request.
        :return: A stream of messages.
        """
        yield None
