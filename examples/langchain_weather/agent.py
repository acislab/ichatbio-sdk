"""
This agent runs as a LangChain tool-calling agent. Such agents call a sequence of tools to try to to fulfill the user's
request. This is modeled as a conversation which begins with just the user's request, then each subsequent tool call
appends agent-generated messages to the conversation. This implementation uses two special tools - "abort" and "finish"
- which the agent calls when it decides that either it has successfully fulfilled the user's request ("finish") or that
it isn't able to do so and should quit instead ("abort").

See the flowchart in README.md for a visualization of the agent.
"""

from typing import override

import langchain.agents
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolRuntime
from pydantic import BaseModel
from starlette.applications import Starlette

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.server import build_agent_app
from ichatbio.types import AgentCard, AgentEntrypoint
from langchain_weather.context import current_context
from langchain_weather.util import context_tool


class LangChainAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="LangChain Example Agent",
            description="A simple example agent that uses LangChain to run a tool-calling loop.",
            icon=None,
            documentation_url=None,
            url=None,
            entrypoints=[
                AgentEntrypoint(
                    id="run",
                    description="Runs this agent.",
                    parameters=None,
                )
            ],
        )

    @override
    async def run(
            self,
            context: ResponseContext,
            request: str,
            entrypoint: str,
            params: BaseModel,  # It's safe to assume type Parameter because we only have one entrypoint
    ):
        """
        Running this agent first builds a LangChain agent graph (a loop that alternates between decision-making and
        tool execution), then executes the graph with `request` as input. The tools themselves are responsible for
        sending messages back to iChatBio via the `context` object. To give the tools access to the request context, we
        instantiate new tools each time a request is received; this allows the agent to safely handle concurrent
        requests.
        """

        current_context.set(context)

        # Build a LangChain agent graph
        agent = langchain.agents.create_agent(
            model=ChatOpenAI(model="gpt-4.1-nano", tool_choice="required"),
            tools=[check_weather, abort, finish],
            system_prompt="You're a friendly weather assistant.",
        )

        # Run the graph

        await agent.ainvoke(
            {
                "messages": [
                    {"role": "user", "content": request},
                ]
            }
        )


@context_tool
async def check_weather(location: str, runtime: ToolRuntime):
    """Reports the current weather conditions at a specified location."""
    context = current_context.get()
    async with context.begin_process("Looking up...") as process:
        process: IChatBioAgentProcess
        await process.log(f"Clear skies in sunny {location}!")


@tool(return_direct=True)  # This tool ends the agent loop
async def abort(reason: str, runtime: ToolRuntime):
    """If you can't fulfill the user's request, abort instead and explain why."""
    await current_context.get().reply(reason)


@tool(return_direct=True)  # This tool ends the agent loop
async def finish(message: str, runtime: ToolRuntime):
    """Mark the user's request as successfully completed."""
    await current_context.get().reply(message)


def create_app() -> Starlette:
    agent = LangChainAgent()
    app = build_agent_app(agent)
    return app
