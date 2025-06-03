from multiprocessing import Process
from time import sleep
from typing import Optional, AsyncGenerator
from uuid import uuid4

import a2a.client
import httpx
import pytest
from a2a.types import MessageSendParams, SendStreamingMessageRequest, AgentCard
from pydantic import BaseModel

import ichatbio.types
from ichatbio.agent import IChatBioAgent
from ichatbio.server import run_agent_server
from ichatbio.types import AgentEntrypoint, Message, TextMessage


@pytest.fixture
def test_agent():
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
                url="http://localhost:9999"
            )

        async def run(self, request: str, entrypoint: str, params: Optional[BaseModel]) -> AsyncGenerator[
            None, Message]:
            yield TextMessage(text="first message")

    return TestAgent()


def run_test_agent_server(test_agent):
    run_agent_server(test_agent, "0.0.0.0", 9999)


@pytest.fixture
def agent_server(test_agent):
    proc = Process(target=run_test_agent_server, args=(test_agent,), daemon=True)
    proc.start()
    sleep(1)  # Whatever
    yield
    proc.kill()


async def query_test_agent(agent_server, message_payload):
    web_client = httpx.AsyncClient(timeout=None)
    a2a_client = a2a.client.A2AClient(web_client, url="http://localhost:9999")

    send_message_payload = {"message": message_payload}

    request = SendStreamingMessageRequest(
        id=str(uuid4()), params=MessageSendParams(**send_message_payload)
    )

    messages = [message async for message in a2a_client.send_message_streaming(request)]
    return messages


@pytest.mark.asyncio
async def test_server(agent_server):
    messages = await query_test_agent(agent_server, {
        "role": "user",
        "parts": [
            {"kind": "text", "text": "Do something for me"},
            {"kind": "data", "data": {"entrypoint": {"id": "no_parameters"}}},
        ],
        "messageId": uuid4().hex,
    })

    assert len(messages) == 4
