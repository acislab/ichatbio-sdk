import json

import a2a.types
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from starlette.applications import Starlette

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_executor import IChatBioAgentExecutor
from ichatbio.types import AgentCard


def convert_agent_card_to_a2a(card: AgentCard):
    return a2a.types.AgentCard(
        name=card.name,
        description=card.description,
        url=str(card.url),
        version="1",
        capabilities=a2a.types.AgentCapabilities(streaming=True),
        defaultInputModes=["text/plain"],
        defaultOutputModes=["text/plain"],
        skills=[a2a.types.AgentSkill(
            id=entrypoint.id,
            name=entrypoint.id,
            description=json.dumps({"description": entrypoint.description} | \
                                   ({"parameters": entrypoint.parameters.model_json_schema()}
                                    if entrypoint.parameters else {})),
            tags=["ichatbio"],
        ) for entrypoint in card.entrypoints],
    )


def build_agent_app(agent: IChatBioAgent) -> Starlette:
    request_handler = DefaultRequestHandler(
        agent_executor=IChatBioAgentExecutor(agent),
        task_store=InMemoryTaskStore(),
    )

    icb_agent_card = agent.get_agent_card()
    a2a_agent_card = convert_agent_card_to_a2a(icb_agent_card)

    return A2AStarletteApplication(
        agent_card=a2a_agent_card,
        http_handler=request_handler
    ).build()


def run_agent_server(agent: IChatBioAgent, host: str, port: int, url: str = None):
    """
    Starts a web server that serves the agent card and accepts agent requests.
    :param agent: The iChatBio agent to receive requests.
    :param host: Web server host.
    :param port: Web server port.
    """
    server = build_agent_app(agent)
    uvicorn.run(server, host=host, port=port)
