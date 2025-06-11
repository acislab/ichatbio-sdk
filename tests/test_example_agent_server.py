from uuid import uuid4

import a2a.client
import httpx
import pytest
import pytest_asyncio
from a2a.types import MessageSendParams, SendStreamingMessageRequest
from httpx import ASGITransport

from examples.cataas.agent import CataasAgent
from ichatbio.server import build_agent_app


@pytest.fixture
def agent():
    yield CataasAgent()


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
                {"kind": "text", "text": "I need a sphynx"},
                {"kind": "data", "data": {"entrypoint": {"id": "get_cat_image"}}},
            ],
            "messageId": uuid4().hex,
        },
    }

    request = SendStreamingMessageRequest(params=MessageSendParams(**send_message_payload))

    messages = [message async for message in agent_a2a_client.send_message_streaming(request)]

    assert len(messages) == 9
