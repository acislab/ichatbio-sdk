import pytest
from pydantic import ValidationError, BaseModel, Field

from ichatbio.types import AgentCard, AgentEntrypoint, Artifact


def test_agent_card_must_have_entrypoints():
    AgentCard(
        name="Test",
        description="Just a test.",
        entrypoints=[
            AgentEntrypoint(id="test", description="Just a test.", parameters=None)
        ],
    )

    with pytest.raises(ValidationError):
        AgentCard(name="Test", description="Just a test.", entrypoints=[])


def test_artifact_schema():
    class ModelWithAnArtifactParameter(BaseModel):
        artifact: Artifact = Field(description="This is just a test")

    schema = ModelWithAnArtifactParameter.model_json_schema()
    assert schema == {
        "properties": {
            "artifact": {
                "description": "This is just a test",
                "examples": ["#0a9f"],
                "pattern": "^#[0-9a-f]{4}$",
                "type": "string",
            }
        },
        "required": ["artifact"],
        "title": "ModelWithAnArtifactParameter",
        "type": "object",
    }


def test_artifact_is_not_an_identifier():
    with pytest.raises(ValidationError):
        Artifact.validate("#0a9f")


def test_artifact_construction():
    Artifact.model_validate(
        {
            "local_id": "#09af",
            "description": "This is a test",
            "mimetype": "text/plain",
            "uris": ["https://some.content", "hash://md5/deadbeef"],
            "metadata": {"just": "a test"},
        }
    )
