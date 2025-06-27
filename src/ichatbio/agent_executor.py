from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import UnsupportedOperationError, TaskState
from a2a.utils import new_agent_text_message
from a2a.utils.errors import ServerError
from pydantic import ValidationError
from typing_extensions import override

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import ResponseContext, ResponseChannel


async def _reject_on_parsing_error(updater: TaskUpdater, exception):
    await updater.update_status(TaskState.rejected, new_agent_text_message(
        "Failed to parse request parameters: " + str(exception)
    ), final=True)


async def _reject_on_unrecognized_entrypoint(updater: TaskUpdater, agent, entrypoint_id):
    await updater.update_status(TaskState.rejected, new_agent_text_message(
        f"Unrecognized entrypoint \"{entrypoint_id}\". Available entrypoints:\n" +
        "[" + ", ".join(e.id for e in agent.get_agent_card().entrypoints) + "]"
    ), final=True)


async def _reject_on_bad_parameters(updater: TaskUpdater, exception):
    await updater.update_status(TaskState.rejected, new_agent_text_message(
        "Request parameters do not match schema: " + str(exception)
    ), final=True)


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
        # Run the agent until either complete or the task is suspended.
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        # Immediately notify that the task is submitted.
        if not context.current_task:
            await updater.submit()

        # TODO: for now, assume messages begin with a text part and a data part
        try:
            request_text: str = context.message.parts[0].root.text
            request_data: dict = context.message.parts[1].root.data

            raw_entrypoint_data = request_data["entrypoint"]
            entrypoint_id = raw_entrypoint_data["id"]
            raw_entrypoint_params = raw_entrypoint_data["parameters"] if "parameters" in raw_entrypoint_data else {}

        except (AttributeError, IndexError, KeyError) as e:
            return await _reject_on_parsing_error(updater, e)

        entrypoint = next((e for e in self.agent.get_agent_card().entrypoints if e.id == entrypoint_id), None)

        if not entrypoint:
            return await _reject_on_unrecognized_entrypoint(updater, self.agent, entrypoint_id)

        if entrypoint.parameters is not None:
            try:
                entrypoint_params = entrypoint.parameters(**raw_entrypoint_params)
            except ValidationError as e:
                return await _reject_on_bad_parameters(updater, e)
        else:
            entrypoint_params = None

        await updater.start_work()

        response_channel = ResponseChannel(context, updater)
        response_context = ResponseContext(response_channel, context.task_id)
        await self.agent.run(response_context, request_text, entrypoint_id, entrypoint_params)

        await updater.complete()

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
