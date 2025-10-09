import os

import pytest

from examples.hello_world.agent import HelloWorldAgent
from ichatbio.agent_response import ProcessLogResponse, ArtifactResponse, ProcessBeginResponse, DirectResponse


@pytest.mark.asyncio
async def test_hello(context, messages):

    agent = HelloWorldAgent()
    await agent.run(context, "Hi", "hello", None)
    assert messages == [
        ProcessBeginResponse("Thinking"),
        ProcessLogResponse("Hello world!"),
        DirectResponse("I said it!")
    ]


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Makes LLM calls")
@pytest.mark.asyncio
async def test_cataas(context, messages):
    from examples.cataas.agent import CataasAgent, GetCatImageParameters

    agent = CataasAgent()
    await agent.run(context, "I need a Sphynx", "get_cat_image", GetCatImageParameters())
    process_log = [m for m in messages if type(m) is ProcessLogResponse]

    assert process_log[:3] == [
        ProcessLogResponse("Generating search parameters"),
        ProcessLogResponse("Search parameters", {"search_parameters": {"tags": ["sphynx"]}}),
        ProcessLogResponse("Sending GET request to https://cataas.com/cat/sphynx")
    ]

    artifacts = [p for p in messages if type(p) is ArtifactResponse]
    assert len(artifacts) == 1

    artifact = artifacts[0]
    assert artifact.mimetype == "image/png"
    assert artifact.content
    assert artifact.metadata == {"api_query_url": "https://cataas.com/cat/sphynx"}


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Makes LLM calls")
@pytest.mark.asyncio
async def test_vision(context, messages):
    from examples.vision.agent import VisionAgent, ExamineParameters

    agent = VisionAgent()
    await agent.run(
        context,
        "In no more than 5 words, what is this?", "find_occurrence_records",
        ExamineParameters(**{
            "image": {
                "local_id": "#0123",
                "description": "A menacing amphibian.",
                "mimetype": "image/jpeg",
                "uris": [
                    "https://media.australian.museum/media/dd/images/Litoria_rothii_NT_calling_Rowley.7050415.width-1600.19ba5f6.jpg"
                ],
                "metadata": {}
            }
        })
    )

    process_messages = [p for p in messages if isinstance(p, ProcessLogResponse)]
    assert len(process_messages) == 3

    ai_analysis = process_messages[-1].text
    assert "frog" in ai_analysis.lower()
