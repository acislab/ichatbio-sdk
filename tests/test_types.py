import pytest
from pydantic import ValidationError

from ichatbio.types import AgentCard, AgentEntrypoint


def test_agent_card_id():
    AgentCard(
        id="this_is_correct123",
        name="Test",
        description="Just a test.",
        entrypoints=[AgentEntrypoint(id="test", description="Just a test.", parameters=None)]
    )

    with pytest.raises(ValidationError):
        AgentCard(
            id="This isn't formatted correctly!",
            name="Test",
            description="Just a test.",
            entrypoints=[AgentEntrypoint(id="test", description="Just a test.", parameters=None)]
        )


def test_agent_card_must_have_entrypoints():
    AgentCard(
        id="test",
        name="Test",
        description="Just a test.",
        entrypoints=[AgentEntrypoint(id="test", description="Just a test.", parameters=None)]
    )

    with pytest.raises(ValidationError):
        AgentCard(
            id="test",
            name="Test",
            description="Just a test.",
            entrypoints=[]
        )
