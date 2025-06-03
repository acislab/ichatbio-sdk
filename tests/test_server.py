from multiprocessing import Process
from uuid import uuid4

import a2a.client
import httpx
import pytest
from a2a.types import MessageSendParams, SendStreamingMessageRequest
from anyio import sleep

from examples.cataas.agent import CataasAgent
from ichatbio.server import run_agent_server


def run_test_agent_server():
    agent = CataasAgent()
    run_agent_server(agent, "0.0.0.0", 9999)


@pytest.fixture
def agent_server():
    proc = Process(target=run_test_agent_server, args=(), daemon=True)
    proc.start()
    yield
    proc.kill()


@pytest.mark.asyncio
async def test_bad_entrypoint(agent_server):
    await sleep(5)
    web_client = httpx.AsyncClient(timeout=None)
    a2a_client = a2a.client.A2AClient(web_client, url="http://localhost:9999")

    send_message_payload = {
        "message": {
            "role": "user",
            "parts": [
                {"kind": "text", "text": "I need a sphynx"}
            ],
            "messageId": uuid4().hex,
        },
    }

    request = SendStreamingMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**send_message_payload)
    )

    messages = [message async for message in a2a_client.send_message_streaming(request)]

    assert len(messages) == 9


@pytest.mark.asyncio
async def test_server(agent_server):
    await sleep(2)
    web_client = httpx.AsyncClient(timeout=None)
    a2a_client = a2a.client.A2AClient(web_client, url="http://localhost:9999")

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

    request = SendStreamingMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**send_message_payload)
    )

    messages = [message async for message in a2a_client.send_message_streaming(request)]

    assert len(messages) == 9
