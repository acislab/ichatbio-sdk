import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from examples.cataas.agent import CataasAgent
from ichatbio.agent_executor import IChatBioAgentExecutor

if __name__ == "__main__":
    agent = CataasAgent()

    request_handler = DefaultRequestHandler(
        agent_executor=IChatBioAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=agent.agent_card, http_handler=request_handler
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=9999)
