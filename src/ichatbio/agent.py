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
        Receives requests from iChatBio. The `context` object is used to send text responses and initiate
        data-generating processes. There are two ways to respond to requests:

        - ``context.reply(message)`` responds directly to iChatBio, not the user. **Only** use this method to converse with iChatBio, such as asking for more information or providing guidance about next steps. **Do not** use this method to transmit data or descriptions of data-generating processes.
        - ``context.begin_process(summary)`` begins a new process which allows the agent to describe its actions and transmit data. It should be used in a `with` statement, e.g.,

            >>> with context.begin_process("Retrieving data") as process:
            >>>     process.log("I found 10KB of data")
            >>>     process.create_artifact(...)

        :param context: Facilitates response interactions with iChatBio.
        :param request: A natural language description of what the agent should do.
        :param entrypoint: The name of the entrypoint selected to handle this request.
        :param params: Request-related information structured according to the entrypoint's parameter data model.
        :return: A stream of messages.
        """
        pass
