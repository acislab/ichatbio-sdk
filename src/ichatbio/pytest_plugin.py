import pytest

from ichatbio.agent_response import ResponseContext
from ichatbio.test_utils import InMemoryResponseChannel


@pytest.fixture(scope="function")
def messages():
    return list()


@pytest.fixture(scope="function")
def context(messages) -> ResponseContext:
    return ResponseContext(InMemoryResponseChannel(messages))
