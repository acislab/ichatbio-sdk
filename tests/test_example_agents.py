import pytest

from examples.cataas.agent import CataasAgent
from ichatbio.types import TextMessage, ArtifactMessage


@pytest.mark.asyncio
async def test_cataas():
    agent = CataasAgent()
    response = agent.run("I need a Sphynx", "get_cat_image", None)
    messages = [m async for m in response]

    m1: TextMessage = messages[0]
    assert type(m1) is TextMessage
    assert m1.text == "Cat retrieved."
    assert m1.data is None

    m2: ArtifactMessage = messages[1]
    assert type(m2) is ArtifactMessage
    assert m2.mimetype == "image/png"
    assert m2.content
    assert m2.metadata == {"api_query_url": "https://cataas.com/cat/sphynx"}
