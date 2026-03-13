import functools
import types
from contextlib import contextmanager

import langchain.tools

from langchain_weather.context import current_context
from ichatbio.agent_response import (
    ArtifactResponse,
    ResponseContext,
)


def context_tool(func):
    """
    Converts a function into a langchain tool that emits iChatBio messages. Use contextvars to get the current iChatBio
    RequestContext or other related non-serializable objects.

    Example use:

    >>> @context_tool
    >>> async def do_something(request: str):
    >>>     context = current_context.get()
    >>>     # Now do something...
    """

    @langchain.tools.tool(func.__name__, description=func.__doc__)
    @functools.wraps(func)  # Preserves function signature
    async def wrapper(*args, **kwargs):
        context = current_context.get()
        with capture_messages(context) as messages:
            await func(*args, **kwargs)
            return messages  # Pass the iChatBio messages back to the LangChain agent as context

    return wrapper


@contextmanager
def capture_messages(context: ResponseContext):
    """
    Modifies a ResponseContext so that any messages sent back to iChatBio are also collected into a list.

    Usage:
        context: ResponseContext
        with capture_messages(context) as messages:
            await context.reply("Alert!")
            # Now messages[0] is a DirectResponse object
    """
    messages = []

    channel = context._channel
    old_submit = channel.submit

    async def submit_and_buffer(self, message):
        await old_submit(message)
        match message:
            case ArtifactResponse() as artifact:
                # Remove "content" from artifact messages, the AI doesn't need to see it:
                # - The artifact description and metadata provide enough context for decision-making
                # - If the AI sees content, it may process it directly instead of running reliable processes
                # - It can be expensive to include content in LLM context, and it might not even fit
                messages.append(
                    ArtifactResponse(
                        description=artifact.description,
                        mimetype=artifact.mimetype,
                        metadata=artifact.metadata,
                    )
                )
            case _:
                messages.append(message)

    channel.submit = types.MethodType(submit_and_buffer, channel)

    yield messages

    channel.submit = old_submit
