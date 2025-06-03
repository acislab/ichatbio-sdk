import base64

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import UnsupportedOperationError, TextPart, Part, DataPart, FilePart, FileWithBytes, FileWithUri, \
    TaskState
from a2a.utils import new_agent_parts_message, new_agent_text_message
from a2a.utils.errors import ServerError
from pydantic import ValidationError
from typing_extensions import override

from ichatbio.agent import IChatBioAgent
from ichatbio.types import ProcessMessage, TextMessage, ArtifactMessage


def reject_on_parsing_error(updater: TaskUpdater, exception):
    updater.update_status(TaskState.rejected, new_agent_text_message(
        "Failed to parse request parameters: " + str(exception)
    ), final=True)


def reject_on_unrecognized_entrypoint(updater: TaskUpdater, agent, entrypoint_id):
    updater.update_status(TaskState.rejected, new_agent_text_message(
        f"Unrecognized entrypoint \"{entrypoint_id}\". Available entrypoints:\n" +
        "[" + ", ".join(e.id for e in agent.get_agent_card().entrypoints) + "]"
    ), final=True)


def reject_on_bad_parameters(updater: TaskUpdater, exception):
    updater.update_status(TaskState.rejected, new_agent_text_message(
        "Request parameters do not match schema: " + str(exception)
    ), final=True)


class IChatBioAgentExecutor(AgentExecutor):
    """Test AgentProxy Implementation."""

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
            updater.submit()

        # TODO: for now, assume messages begin with a text part and a data part
        try:
            request_text: str = context.message.parts[0].root.text
            request_data: dict = context.message.parts[1].root.data

            raw_entrypoint_data = request_data["entrypoint"]
            entrypoint_id = raw_entrypoint_data["id"]
            raw_entrypoint_params = raw_entrypoint_data["parameters"] if "parameters" in raw_entrypoint_data else {}

        except (AttributeError, IndexError, KeyError) as e:
            return reject_on_parsing_error(updater, e)

        entrypoint = next((e for e in self.agent.get_agent_card().entrypoints if e.id == entrypoint_id), None)

        if not entrypoint:
            return reject_on_unrecognized_entrypoint(entrypoint_id, self.agent)

        if entrypoint.parameters is not None:
            try:
                entrypoint_params = entrypoint.parameters(**raw_entrypoint_params)
            except ValidationError as e:
                return reject_on_bad_parameters(updater, e)
        else:
            entrypoint_params = None

        updater.start_work()

        async for message in self.agent.run(request_text, entrypoint_id, entrypoint_params):
            match message:
                case ProcessMessage(summary=summary, description=description, data=data):
                    parts = [DataPart(data={
                        "summary": summary,
                        "description": description,
                        "data": data
                    })]

                case TextMessage(text=text, data=data):
                    parts = [TextPart(text=text, metadata=data)]

                case ArtifactMessage(uris=uris, content=content, mimetype=mimetype, metadata=metadata,
                                     description=description):
                    if content:
                        file = FileWithBytes(
                            bytes=base64.b64encode(content),
                            mimeType=mimetype,
                            name=description
                        )
                    elif uris:
                        file = FileWithUri(
                            uri=uris[0],
                            mimeType=mimetype,
                            name=description
                        )
                    else:
                        raise ValueError("Artifact message must have at least one URI or non-empty content")

                    parts = [FilePart(
                        file=file,
                        metadata={
                            "uris": uris,
                            "metadata": metadata
                        }
                    )]

                case _:
                    raise ValueError("Outgoing messages must be of type ProcessMessage | TextMessage | ArtifactMessage")

            updater.update_status(
                TaskState.working,
                new_agent_parts_message(
                    [Part(root=p) for p in parts],
                    context.context_id,
                    context.task_id)
            )

        updater.complete()

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
