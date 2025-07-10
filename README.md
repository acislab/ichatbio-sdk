# iChatBio SDK

[![tests](https://github.com/acislab/ichatbio-sdk/actions/workflows/tests.yml/badge.svg)](https://github.com/acislab/ichatbio-sdk/actions/workflows/tests.yml)

The iChatBio SDK is designed to aid in the development of agents that can communicate with iChatBio. The SDK adds a
layer of abstraction over the [A2A protocol](https://github.com/google/a2a), hiding the complexities of A2A while
exposing iChatBio-specific capabilities. Because agents designed with the iChatBio SDK make use of A2A, they are also
able to communicate with other A2A agents, though without access to services (e.g., strict data models, special messages
types, and shared persistent storage) enabled by the iChatBio ecosystem.

# Getting started

See [examples](examples) for a reference agent implementation. A standalone example agent is
available [here](https://github.com/mielliott/ichatbio-agent-example).

The iChatBio SDK is available on PyPI:

```sh
pip install ichatbio-sdk
```

Like A2A, iChatBio agents must define an agent card. Here's an example card:

```python
from ichatbio.types import AgentCard, AgentEntrypoint

card = AgentCard(
    name="Friendly Agent",
    description="Responds in a friendly manner.",
    icon="https://example.com/icon.png",
    url="http://localhost:9999",
    entrypoints=[
        AgentEntrypoint(
            id="chat",
            description="Generates a friendly reply.",
            parameters=Parameters  # Defined below
        )
    ]
)
```

The card must include one or more *entrypoints*. Entrypoints define the types of interactions that are possible between
iChatBio and the agent. Each entrypoint can optionally define a set of *parameters*, which allow iChatBio to provide
structured information to the agent. This structure has a number of advantages:

* Agents can directly access parameters without the unreliable overhead of natural language processing
* Agents with strict parameter sets can only be used when the required parameters are supplied

Here's the parameter model referenced in the entrypoint above:

```python
from pydantic import BaseModel, PastDate


class Parameters(BaseModel):
    birthday: PastDate
```

By using Pydantic's `PastDate` class, the birthday must both be a valid date and also be a date *in the past*. With
these constraints, the agent does not need to worry about receiving invalid parameter values and subsequent error
handling.

Here's an agent that implements the `"chat"` entrypoint:

```python
from datetime import date
from typing import override

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, IChatBioAgentProcess
from ichatbio.types import AgentCard


class FriendlyAgent(IChatBioAgent):
    @override
    def get_agent_card(self) -> AgentCard:
        return card  # The AgentCard we defined earlier

    @override
    async def run(self, context: ResponseContext, request: str, entrypoint: str, params: Parameters):
        if entrypoint != "chat":
            raise ValueError()  # This should never happen

        async with context.begin_process(summary="Replying") as process:
            process: IChatBioAgentProcess

            await process.log("Generating a friendly reply")
            response = ...  # Query an LLM

            await process.log("Response generated", data={"response": response})

            happy_birthday = params.birthday == date.today()
            if happy_birthday:
                await process.log("Generating a birthday surprise")
                audio: bytes = ...  # Generate an audio version of the response
                await process.create_artifact(
                    mimetype="audio/mpeg",
                    description=f"An audio version of the response",
                    content=audio
                )

            await context.reply(
                "I have generated a friendly response to the user's request. For their birthday, I also generated an"
                " audio version of the response."
                if happy_birthday else
                "I have generated a friendly response to the user's request."
            )
```

And here's a `__main__.py` to run the agent as an A2A web server:

```python
from ichatbio.server import run_agent_server

if __name__ == "__main__":
    agent = FriendlyAgent()
    run_agent_server(agent, host="0.0.0.0", port=9999)
```

If all went well, you should be able to find your agent card at http://localhost:9999/.well-known/agent.json.

# SDK Development

Requires Python 3.12 or higher.

Dependencies for the example agents are installed separately:

```
pip install .[example]
```

# Funding

This work is funded by grants from the National Science Foundation (DBI 2027654) and the AT&T Foundation.
