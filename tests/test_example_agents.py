import pytest

import examples.vision.agent
from examples.cataas.agent import CataasAgent, GetCatImageParameters
from examples.hello_world.agent import HelloWorldAgent
from examples.idigbio.agent import IDigBioAgent
from examples.vision.agent import VisionAgent
from ichatbio.agent_response import ProcessLogResponse, ArtifactResponse, ProcessBeginResponse, TextResponse


@pytest.mark.asyncio
async def test_hello(context, messages):
    agent = HelloWorldAgent()
    await agent.run(context, "Hi", "hello", None)
    assert messages == [
        ProcessBeginResponse("Thinking"),
        ProcessLogResponse("Hello world!"),
        TextResponse("I said it!")
    ]


@pytest.mark.asyncio
async def test_cataas(context, messages):
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


@pytest.mark.asyncio
async def test_idigbio(context, messages):
    agent = IDigBioAgent()
    await agent.run(context, "Retrieve records of Rattus rattus in Alaska", "find_occurrence_records", None)

    process_messages = [p for p in messages if isinstance(p, ProcessLogResponse)]
    assert len(process_messages) == 5

    search_parameters = next(p.data for p in process_messages if p.text == "Generated search parameters")
    assert search_parameters == {
        "limit": 100,
        "rq": {
            "scientificname": "Rattus rattus",
            "stateprovince": "Alaska"
        }
    }

    artifacts = [p for p in messages if type(p) is ArtifactResponse]
    assert len(artifacts) == 1

    artifact: ArtifactResponse = artifacts[0]
    assert artifact.mimetype == "application/json"
    assert artifact.uris
    assert artifact.metadata["data_source"] == "iDigBio"


@pytest.mark.asyncio
async def test_vision(context, messages):
    agent = VisionAgent()
    await agent.run(
        context,
        "In no more than 5 words, what is this?", "find_occurrence_records",
        examples.vision.agent.ExamineParameters(**{
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
