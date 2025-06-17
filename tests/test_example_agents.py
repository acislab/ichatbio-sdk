import pytest

import examples.vision.agent
from examples.cataas.agent import CataasAgent, GetCatImageParameters
from examples.idigbio.agent import IDigBioAgent
from examples.vision.agent import VisionAgent
from ichatbio.types import ArtifactMessage, ProcessMessage


@pytest.mark.asyncio
async def test_cataas():
    agent = CataasAgent()
    response = agent.run("I need a Sphynx", "get_cat_image", GetCatImageParameters())
    messages = [m async for m in response]

    process_summaries = [p.summary for p in messages if type(p) is ProcessMessage and p.summary]

    assert process_summaries == [
        "Searching for cats",
        "Retrieving cat",
        "Cat retrieved"
    ]

    artifacts = [p for p in messages if type(p) is ArtifactMessage]
    assert len(artifacts) == 1

    artifact = artifacts[0]
    assert artifact.mimetype == "image/png"
    assert artifact.content
    assert artifact.metadata == {"api_query_url": "https://cataas.com/cat/sphynx"}


@pytest.mark.asyncio
async def test_idigbio():
    agent = IDigBioAgent()
    response = agent.run("Retrieve records of Rattus rattus in Alaska", "find_occurrence_records", None)
    messages = [m async for m in response]

    process_messages: list[ProcessMessage] = [p for p in messages if type(p) is ProcessMessage]
    assert len(process_messages) == 5

    search_parameters = next(p.data for p in process_messages if p.description == "Generated search parameters")
    assert search_parameters == {
        "limit": 100,
        "rq": {
            "scientificname": "Rattus rattus",
            "stateprovince": "Alaska"
        }
    }

    artifacts = [p for p in messages if type(p) is ArtifactMessage]
    assert len(artifacts) == 1

    artifact: ArtifactMessage = artifacts[0]
    assert artifact.mimetype == "application/json"
    assert artifact.uris
    assert artifact.metadata["data_source"] == "iDigBio"


@pytest.mark.asyncio
async def test_vision():
    agent = VisionAgent()
    response = agent.run("In no more than 5 words, what is this?", "find_occurrence_records",
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
                         }))
    messages = [m async for m in response]

    process_messages: list[ProcessMessage] = [p for p in messages if type(p) is ProcessMessage]
    assert len(process_messages) == 3

    ai_analysis = process_messages[-1].description
    assert "frog" in ai_analysis.lower()
