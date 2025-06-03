from multiprocessing import Process
from uuid import uuid4

import a2a.client
import httpx
import pytest
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import SendMessageRequest, MessageSendParams, AgentCapabilities, SendStreamingMessageRequest
from anyio import sleep

from examples.cataas.agent import CataasAgent
from ichatbio.agent_executor import IChatBioAgentExecutor


def run_agent_server():
    agent = CataasAgent()

    request_handler = DefaultRequestHandler(
        agent_executor=IChatBioAgentExecutor(agent),
        task_store=InMemoryTaskStore(),
    )

    icb_agent_card = agent.get_agent_card()
    a2a_agent_card = a2a.types.AgentCard(
        name=icb_agent_card.name,
        description=icb_agent_card.description,
        skills=[a2a.types.AgentSkill(
            id=entrypoint.id,
            name=entrypoint.id,
            description=entrypoint.description,
            tags=["ichatbio"]
        ) for entrypoint in icb_agent_card.entrypoints],
        url="http://localhost:9999",
        version="1",
        capabilities=AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"]
    )

    server = A2AStarletteApplication(
        agent_card=a2a_agent_card, http_handler=request_handler
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=9999)


@pytest.fixture
def agent_server():
    proc = Process(target=run_agent_server, args=(), daemon=True)
    proc.start()
    yield
    proc.kill()


@pytest.mark.asyncio
async def test_server(agent_server):
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

    pass
