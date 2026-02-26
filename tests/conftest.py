import pytest

from ichatbio.agent_response import ResponseChannel, ResponseContext, ResponseMessage


class InMemoryResponseChannel(ResponseChannel):
    """
    Useful for interacting with agents locally (e.g., unit tests, command line interfaces) instead of sending responses
    over the network. The `message_buffer` is populated by running an agent.

    Example:

        messages = list()
        channel = InMemoryResponseChannel(messages)
        context = ResponseContext(channel)

        # `messages` is empty
        agent = HelloWorldAgent()
        await agent.run(context, "Hi", "hello", None)

        # `messages` should now be populated
        assert messages[0].text == "Hello world!"
    """

    def __init__(self, message_buffer: list):
        super().__init__()
        self.message_buffer = message_buffer

    async def submit(self, message: ResponseMessage):
        self.message_buffer.append(message)


@pytest.fixture(scope="function")
def messages():
    return list()


@pytest.fixture(scope="function")
def context(messages) -> ResponseContext:
    return ResponseContext(InMemoryResponseChannel(messages))
