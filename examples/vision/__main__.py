from ichatbio.server import run_agent_server
from .agent import VisionAgent

if __name__ == "__main__":
    agent = VisionAgent()
    run_agent_server(agent, host="0.0.0.0", port=9999)
