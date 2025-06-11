import pytest
from pydantic import ValidationError

from ichatbio.types import AgentCard, AgentEntrypoint


def test_agent_card_must_have_entrypoints():
    AgentCard(
        name="Test",
        description="Just a test.",
        entrypoints=[AgentEntrypoint(id="test", description="Just a test.", parameters=None)]
    )

    with pytest.raises(ValidationError):
        AgentCard(
            name="Test",
            description="Just a test.",
            entrypoints=[]
        )
