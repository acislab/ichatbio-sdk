from contextvars import ContextVar

from ichatbio.agent_response import ResponseContext

current_context: ContextVar[ResponseContext] = ContextVar("current_context")
