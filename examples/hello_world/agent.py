from typing import override, Optional

from pydantic import BaseModel

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard, AgentEntrypoint


class HelloWorldAgent(IChatBioAgent):
    """
    A simple example agent with a single entrypoint.
    """

    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="The Simplest Agent",
            description="Can only say \"Hello world!\".",
            icon="https://commons.wikimedia.org/wiki/Category:Hello_World#/media/File:Qt_example.png",
            url=None,
            entrypoints=[
                AgentEntrypoint(
                    id="hello",
                    description="Responds with \"Hello world!\".",
                    parameters=None
                )
            ]
        )

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Optional[BaseModel]):
        # Start a process to log the agent's actions
        async with context.begin_process(summary="Thinking") as process:
            process: IChatBioAgentProcess

            # Perform any long-running work in this process block, logging steps taken and their outcomes. iChatBio
            # users can see these log messages.
            await process.log("Hello world!")

            # Any uncaught exceptions will trigger a failure response. This is OK! It is often better for an agent
            # to fail when it can't handle a request, rather pretend it was handled successfully.
            if request == "Please fail":
                raise ValueError("This is an exception!")

        # Reply directly to the iChatBio agent, not the user
        await context.reply("I said it!")
