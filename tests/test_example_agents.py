import pytest

from examples.cataas.agent import CataasAgent, GetCatImageParameters
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
