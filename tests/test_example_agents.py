from examples.cataas.agent import CataasAgent
from ichatbio.types import TextMessage, ArtifactMessage


async def test_cataas():
    agent = CataasAgent()
    messages = [m async for m in agent.run("I need a Sphynx", {})]

    m = messages[0]
    assert type(m) is TextMessage
    assert m.text == "Cat retrieved."
    assert m.data is None

    m = messages[1]
    assert type(m) is ArtifactMessage
    assert m.mimetype == "image/png"
    assert m.content
    assert m.metadata == {"api_query_url": "https://cataas.com/cat/sphynx"}
