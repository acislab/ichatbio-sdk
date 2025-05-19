from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import UnsupportedOperationError, TextPart, Part, DataPart, FilePart, FileWithBytes, FileWithUri
from a2a.utils import new_agent_parts_message
from a2a.utils.errors import ServerError
from typing_extensions import override

from ichatbio.agent import IChatBioAgent
from ichatbio.types import ProcessMessage, TextMessage, ArtifactMessage


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

        updater.start_work()

        # TODO: for now, take request text from the first TextPart
        first_text_part = next((p for p in context.message.parts if type(p) is TextPart))
        request_text = first_text_part.text

        # TODO: for now, take request parameters from the first DataPart
        first_data_part = next((p for p in context.message.parts if type(p) is DataPart))
        request_params = first_data_part.data

        # TODO: receive conversation, artifacts catalog, etc.
        conversation_context = object()

        async for message in self.agent.run(request_text, request_params):
            match message:
                case ProcessMessage(summary, description, data):
                    parts = [DataPart(data={
                        "summary": summary,
                        "description": description,
                        "data": data
                    })]

                case TextMessage(text, data):
                    parts = [TextPart(text=text, metadata=data)]

                case ArtifactMessage(uris, content, mimetype, metadata, description):
                    if content:
                        file = FileWithBytes(
                            data=content,
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

            event_queue.enqueue_event(
                new_agent_parts_message(
                    [Part(root=p) for p in parts],
                    context.context_id,
                    context.task_id)
            )

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
