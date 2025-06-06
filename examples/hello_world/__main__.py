from examples.hello_world.agent import HelloWorldAgent
from ichatbio.server import run_agent_server

if __name__ == "__main__":
    agent = HelloWorldAgent()
    run_agent_server(agent, host="0.0.0.0", port=9999)
