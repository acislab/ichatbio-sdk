from uuid import uuid4

import a2a.client
import httpx
import pytest
import pytest_asyncio
from a2a.types import MessageSendParams, SendStreamingMessageRequest, TaskState
from httpx import ASGITransport

from hello_world.agent import HelloWorldAgent
from ichatbio.server import build_agent_app


@pytest.fixture
def agent():
    yield HelloWorldAgent()


@pytest_asyncio.fixture
async def agent_httpx_client(agent):
    app = build_agent_app(agent)
    transport = ASGITransport(app)
    async with httpx.AsyncClient(transport=transport) as httpx_client:
        yield httpx_client


@pytest_asyncio.fixture
async def agent_a2a_client(agent_httpx_client):
    return a2a.client.A2AClient(agent_httpx_client, url="http://test.agent")


@pytest.mark.asyncio
async def test_server(agent_a2a_client):
    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [
                {"kind": "text", "text": "hello"},
                {"kind": "data", "data": {"entrypoint": {"id": "hello"}}},
            ],
            "messageId": str(uuid4()),
        }
    }

    request = SendStreamingMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))

    messages = [message async for message in agent_a2a_client.send_message_streaming(request)]
    assert messages[-1].root.result.status.state == TaskState.completed
