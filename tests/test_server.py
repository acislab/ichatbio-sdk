from typing import Optional, AsyncGenerator
from uuid import uuid4

import a2a.client
import httpx
import pytest
import pytest_asyncio
from a2a.types import MessageSendParams, SendStreamingMessageRequest, AgentCard, TaskState
from httpx import ASGITransport
from pydantic import BaseModel

import ichatbio.types
from ichatbio.agent import IChatBioAgent
from ichatbio.server import build_agent_app
from ichatbio.types import AgentEntrypoint, Message, TextMessage

AGENT_URL = "http://localhost:9999"


@pytest.fixture
def agent():
    class OptionalParameters(BaseModel):
        test_parameter: Optional[int] = None

    class StrictParameters(BaseModel):
        test_parameter: int

    class TestAgent(IChatBioAgent):
        def get_agent_card(self) -> AgentCard:
            return ichatbio.types.AgentCard(
                name="test",
                description="test",
                icon=None,
                entrypoints=[
                    AgentEntrypoint(id="no_parameters", description="test", parameters=None),
                    AgentEntrypoint(id="optional_parameters", description="test", parameters=OptionalParameters),
                    AgentEntrypoint(id="strict_parameters", description="test", parameters=StrictParameters),
                ],
                url=AGENT_URL
            )

        async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[
            None, Message]:
            yield TextMessage(text="first message")

    return TestAgent()


@pytest_asyncio.fixture
async def agent_httpx_client(agent):
    app = build_agent_app(agent)
    transport = ASGITransport(app)
    async with httpx.AsyncClient(transport=transport) as httpx_client:
        yield httpx_client


@pytest_asyncio.fixture
async def agent_a2a_client(agent_httpx_client):
    return a2a.client.A2AClient(agent_httpx_client, url="http://test.agent")


async def query_test_agent(agent_a2a_client, message_payload):
    send_message_payload = {"message": message_payload}

    request = SendStreamingMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**send_message_payload)
    )

    messages = [m async for m in agent_a2a_client.send_message_streaming(request)]
    return messages


@pytest.mark.asyncio
async def test_server(agent_a2a_client):
    messages = await query_test_agent(agent_a2a_client, {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Do something for me"},
            {"kind": "data", "data": {"entrypoint": {"id": "no_parameters"}}},
        ],
        "messageId": uuid4().hex,
    })

    assert len(messages) == 4


@pytest.mark.asyncio
async def test_strict_parameters(agent_a2a_client):
    messages = await query_test_agent(agent_a2a_client, {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Do something for me"},
            {"kind": "data", "data": {"entrypoint": {
                "id": "strict_parameters",
                "parameters": {"test_parameter": 1}
            }}},
        ],
        "messageId": uuid4().hex,
    })

    assert len(messages) == 4


@pytest.mark.asyncio
async def test_missing_strict_parameters(agent_a2a_client):
    messages = await query_test_agent(agent_a2a_client, {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Do something for me"},
            {"kind": "data", "data": {"entrypoint": {"id": "strict_parameters"}}},
        ],
        "messageId": uuid4().hex,
    })

    assert messages[-1].root.result.status.state == TaskState.rejected


@pytest.mark.asyncio
async def test_optional_parameters(agent_a2a_client):
    messages = await query_test_agent(agent_a2a_client, {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Do something for me"},
            {"kind": "data", "data": {"entrypoint": {
                "id": "optional_parameters",
                "parameters": {"test_parameter": 1}
            }}},
        ],
        "messageId": uuid4().hex,
    })

    assert len(messages) == 4


@pytest.mark.asyncio
async def test_missing_optional_parameters(agent_a2a_client):
    messages = await query_test_agent(agent_a2a_client, {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Do something for me"},
            {"kind": "data", "data": {"entrypoint": {
                "id": "optional_parameters"
            }}},
        ],
        "messageId": uuid4().hex,
    })

    assert len(messages) == 4


@pytest.mark.asyncio
async def test_bad_parameters(agent_a2a_client):
    messages = await query_test_agent(agent_a2a_client, {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Do something for me"},
            {"kind": "data", "data": {"entrypoint": {
                "id": "strict_parameters",
                "parameters": {"test_parameter": "this is not an integer!"}
            }}},
        ],
        "messageId": uuid4().hex,
    })

    assert messages[-1].root.result.status.state == TaskState.rejected


@pytest.mark.asyncio
async def test_bad_entrypoint(agent_a2a_client):
    messages = await query_test_agent(agent_a2a_client, {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Do something for me"},
            {"kind": "data", "data": {"entrypoint": {
                "id": "this_is_not_real"
            }}},
        ],
        "messageId": uuid4().hex,
    })

    assert messages[-1].root.result.status.state == TaskState.rejected


@pytest.mark.asyncio
async def test_server_agent_card(agent_httpx_client):
    response = await agent_httpx_client.get(f"{AGENT_URL}/.well-known/agent.json")
    card = response.json()

    assert card == {
        'capabilities': {'streaming': True},
        'defaultInputModes': ['text/plain'],
        'defaultOutputModes': ['text/plain'],
        'description': 'test',
        'name': 'test',
        'skills': [
            {
                'description': '{"description": "test"}',
                'id': 'no_parameters',
                'name': 'no_parameters',
                'tags': ['ichatbio']},
            {
                'description': '{"description": "test", "parameters": {"properties": {"test_parameter": {"anyOf": [{"type": "integer"}, {"type": "null"}], "default": null, "title": "Test Parameter"}}, "title": "OptionalParameters", "type": "object"}}',
                'id': 'optional_parameters',
                'name': 'optional_parameters',
                'tags': ['ichatbio']},
            {
                'description': '{"description": "test", "parameters": {"properties": {"test_parameter": {"title": "Test Parameter", "type": "integer"}}, "required": ["test_parameter"], "title": "StrictParameters", "type": "object"}}',
                'id': 'strict_parameters',
                'name': 'strict_parameters',
                'tags': ['ichatbio']
            }
        ],
        'url': AGENT_URL,
        'version': '1'
    }
