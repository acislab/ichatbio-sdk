import logging
import traceback

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part
from a2a.types import UnsupportedOperationError, TaskState, TextPart
from a2a.utils.errors import ServerError
from attr import dataclass
from pydantic import ValidationError
from typing_extensions import override

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, ResponseChannel


@dataclass
class _Request:
    text: str
    data: dict


class BadRequest(ValueError):
    pass


class IChatBioAgentExecutor(AgentExecutor):
    """
    Translates incoming A2A requests into validated agent run parameters, runs the agent, translates outgoing iChatBio messages into A2A task updates to respond to the client's request.

    Invalid requests (missing information, unrecognized entrypoint, bad entrypoint arguments) are rejected immediately without involving the agent.
    """

    def __init__(self, agent: IChatBioAgent):
        self.agent = agent

    @override
    async def execute(
            self,
            context: RequestContext,
            event_queue: EventQueue,
    ) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # Acknowledge the request
        if not context.current_task:
            await updater.submit()

        # Process the request
        request = None
        try:
            if context.message is None:
                raise BadRequest("Request does not contain a message")

            try:
                # TODO: for now, assume messages begin with a text part and a data part
                request = _Request(context.message.parts[0].root.text, context.message.parts[1].root.data)
            except (IndexError, AttributeError) as e:
                raise BadRequest("Request is not formatted as expected") from e

            try:
                raw_entrypoint_data = request.data["entrypoint"]
                entrypoint_id = raw_entrypoint_data["id"]
                raw_entrypoint_params = raw_entrypoint_data["parameters"] if "parameters" in raw_entrypoint_data else {}

            except (AttributeError, IndexError, KeyError) as e:
                raise BadRequest("Failed to parse request data") from e

            entrypoint = next((e for e in self.agent.get_agent_card().entrypoints if e.id == entrypoint_id), None)

            if not entrypoint:
                raise BadRequest(f"Invalid entrypoint \"{entrypoint_id}\"")

            if entrypoint.parameters is not None:
                try:
                    entrypoint_params = entrypoint.parameters(**raw_entrypoint_params)
                except ValidationError as e:
                    raise BadRequest(
                        f"Invalid arguments for entrypoint \"{entrypoint_id}\": {raw_entrypoint_params}"
                    ) from e
            else:
                entrypoint_params = None

            await updater.start_work()

            response_channel = ResponseChannel(context, updater)
            response_context = ResponseContext(response_channel, context.task_id)

            # Pass the request to the agent
            try:
                logging.info(f"Accepting request {request}")
                await self.agent.run(response_context, request.text, entrypoint_id, entrypoint_params)
                await updater.complete()

            # If the agent failed to process the request, mark the task as "failed"
            except Exception as e:
                logging.error(f"An exception was raised while handling request {request}", exc_info=e)
                message = updater.new_agent_message([Part(root=TextPart(text=traceback.format_exc(limit=0)))])
                await updater.update_status(TaskState.failed, message, final=True)

        # If something is wrong with the request, mark the task as "rejected"
        except BadRequest as e:
            logging.warning(f"Rejecting incoming request: {request}", exc_info=e)
            message = updater.new_agent_message([Part(root=TextPart(
                text=f"Request rejected. Reason: {traceback.format_exc(limit=0)}"
            ))])
            await updater.update_status(TaskState.rejected, message, final=True)

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
