import types
from typing import Optional
from uuid import uuid4

import a2a.client
import httpx
import pytest
import pytest_asyncio
from a2a.types import (
    MessageSendParams,
    SendStreamingMessageRequest,
    AgentCard,
    TaskState,
    SendStreamingMessageResponse,
    Message,
    TextPart,
    DataPart,
)
from httpx import ASGITransport
from pydantic import BaseModel

import ichatbio.types
from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext
from ichatbio.server import build_agent_app
from ichatbio.types import AgentEntrypoint


@pytest.fixture
def agent():
    class OptionalParameters(BaseModel):
        test_parameter: Optional[int] = None

    class StrictParameters(BaseModel):
        test_parameter: int

    class TestAgent(IChatBioAgent):
        def get_agent_card(self) -> AgentCard:
            return ichatbio.types.AgentCard(
                name="Test Name",
                description="Test description.",
                icon=None,
                entrypoints=[
                    AgentEntrypoint(
                        id="no_parameters",
                        description="Test description.",
                        parameters=None,
                    ),
                    AgentEntrypoint(
                        id="optional_parameters",
                        description="Test description.",
                        parameters=OptionalParameters,
                    ),
                    AgentEntrypoint(
                        id="strict_parameters",
                        description="Test description.",
                        parameters=StrictParameters,
                    ),
                ],
                url="http://test.agent/",
            )

        async def run(
            self,
            context: ResponseContext,
            request: str,
            entrypoint: str,
            params: Optional[BaseModel],
        ):
            await context.reply("first message")

    return TestAgent()


@pytest_asyncio.fixture(scope="function")
async def agent_httpx_client(agent):
    app = build_agent_app(agent)
    transport = ASGITransport(app)
    async with httpx.AsyncClient(transport=transport) as httpx_client:
        yield httpx_client


@pytest.fixture(scope="function")
def query_test_agent(agent_httpx_client):
    client = a2a.client.A2AClient(agent_httpx_client, url="http://test.agent")

    async def query(message: Message) -> list[SendStreamingMessageResponse]:
        request = SendStreamingMessageRequest(
            id=str(uuid4()), params=MessageSendParams(message=message)
        )

        messages = [m async for m in client.send_message_streaming(request)]
        return messages

    return query


@pytest.mark.asyncio
async def test_server(query_test_agent):
    messages = await query_test_agent(
        Message(
            message_id="message-1",
            role="user",
            parts=[
                TextPart(text="Do something for me"),
                DataPart(data={"entrypoint": {"id": "no_parameters"}}),
            ],
        )
    )

    assert messages[0].root.result.status.state == TaskState.submitted

    assert len(messages) >= 3
    for m in messages[1:-1]:
        assert m.root.result.status.state == TaskState.working

    assert messages[-1].root.result.status.state == TaskState.completed


@pytest.mark.asyncio
async def test_strict_parameters(query_test_agent):
    messages = await query_test_agent(
        Message(
            message_id="message-1",
            role="user",
            parts=[
                TextPart(text="Do something for me"),
                DataPart(
                    data={
                        "entrypoint": {
                            "id": "strict_parameters",
                            "parameters": {"test_parameter": 1},
                        }
                    }
                ),
            ],
        )
    )

    assert messages[-1].root.result.status.state == TaskState.completed


@pytest.mark.asyncio
async def test_missing_strict_parameters(query_test_agent):
    messages = await query_test_agent(
        Message(
            message_id="message-1",
            role="user",
            parts=[
                TextPart(text="Do something for me"),
                DataPart(data={"entrypoint": {"id": "strict_parameters"}}),
            ],
        )
    )

    assert messages[-1].root.result.status.state == TaskState.rejected


@pytest.mark.asyncio
async def test_optional_parameters(query_test_agent):
    messages = await query_test_agent(
        Message(
            message_id="message-1",
            role="user",
            parts=[
                TextPart(text="Do something for me"),
                DataPart(
                    data={
                        "entrypoint": {
                            "id": "optional_parameters",
                            "parameters": {"test_parameter": 1},
                        }
                    }
                ),
            ],
        )
    )

    assert messages[-1].root.result.status.state == TaskState.completed


@pytest.mark.asyncio
async def test_missing_optional_parameters(query_test_agent):
    messages = await query_test_agent(
        Message(
            message_id="message-1",
            role="user",
            parts=[
                TextPart(text="Do something for me"),
                DataPart(data={"entrypoint": {"id": "optional_parameters"}}),
            ],
        )
    )

    assert messages[-1].root.result.status.state == TaskState.completed


@pytest.mark.asyncio
async def test_bad_parameters(query_test_agent):
    messages = await query_test_agent(
        Message(
            message_id="message-1",
            role="user",
            parts=[
                TextPart(text="Do something for me"),
                DataPart(
                    data={
                        "entrypoint": {
                            "id": "strict_parameters",
                            "pparameters": {
                                "test_parameter": "this is not an integer!"
                            },
                        }
                    }
                ),
            ],
        )
    )

    assert messages[-1].root.result.status.state == TaskState.rejected


@pytest.mark.asyncio
async def test_bad_entrypoint(query_test_agent):
    messages = await query_test_agent(
        Message(
            message_id="message-1",
            role="user",
            parts=[
                TextPart(text="Do something for me"),
                DataPart(data={"entrypoint": {"id": "this_is_not_real"}}),
            ],
        )
    )

    assert messages[-1].root.result.status.state == TaskState.rejected


@pytest.mark.asyncio
async def test_server_agent_card(agent_httpx_client):
    response = await agent_httpx_client.get(f"http://test.agent/.well-known/agent.json")
    card = response.json()

    assert card == {
        "capabilities": {
            "extensions": [{
                "description": "Enables iChatBio-specific interactions",
                "required": True,
                "uri": "https://github.com/acislab/ichatbio-sdk/a2a/v1",
                "params": {
                    "entrypoints": [
                        {"description": "Test description.", "id": "no_parameters", "parameters": None},
                        {"description": "Test description.", "id": "optional_parameters", "parameters": {
                            "properties": {
                                "test_parameter": {
                                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                                    "default": None,
                                    "title": "Test Parameter"
                                }
                            },
                            "title": "OptionalParameters",
                            "type": "object"
                        }},
                        {"description": "Test description.", "id": "strict_parameters", "parameters": {
                            "properties": {
                                "test_parameter": {
                                    "title": "Test Parameter",
                                    "type": "integer"
                                }
                            },
                            "required": ["test_parameter"],
                            "title": "StrictParameters",
                            "type": "object"
                        }}
                    ]
                }
            }],
            "streaming": True
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "description": "Test description.",
        "name": "Test Name",
        "preferredTransport": "JSONRPC",
        "protocolVersion": "0.3.0",
        "skills": [
            {
                "description": '{"description": "Test description."}',
                "id": "no_parameters",
                "name": "no_parameters",
                "tags": ["ichatbio"],
            },
            {
                "description": '{"description": "Test description.", "parameters": {"properties": {"test_parameter": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": null, "title": "Test Parameter"}}, "title": "OptionalParameters", "type": "object"}}',
                "id": "optional_parameters",
                "name": "optional_parameters",
                "tags": ["ichatbio"],
            },
            {
                "description": '{"description": "Test description.", "parameters": {"properties": {"test_parameter": {"title": "Test Parameter", "type": "integer"}}, "required": ["test_parameter"], "title": "StrictParameters", "type": "object"}}',
                "id": "strict_parameters",
                "name": "strict_parameters",
                "tags": ["ichatbio"],
            },
        ],
        "url": "http://test.agent/",
        "version": "0",
    }


@pytest.mark.asyncio
async def test_server(agent, query_test_agent):
    async def explode(*args, **kwargs):
        raise ValueError("Rut roh!")

    agent.run = types.MethodType(explode, agent)

    messages = await query_test_agent(
        Message(
            message_id="message-1",
            role="user",
            parts=[
                TextPart(text="Do something for me"),
                DataPart(data={"entrypoint": {"id": "no_parameters"}}),
            ],
        )
    )

    assert messages[-1].root.result.status.state == TaskState.failed
