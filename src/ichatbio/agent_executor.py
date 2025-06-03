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

        # TODO: for now, take request text from the first TextPart
        first_text_part = next((p.root for p in context.message.parts if isinstance(p.root, TextPart)))
        request_text = first_text_part.text

        # TODO: for now, take request parameters from the first DataPart
        first_data_part = next((p.root for p in context.message.parts if isinstance(p.root, DataPart)), None)
        request_params = first_data_part.data if first_data_part else None

        entrypoint_id = "get_cat_image"
        entrypoint = next((e for e in self.agent.get_agent_card().entrypoints if e.id == entrypoint_id), None)

        if entrypoint and entrypoint.parameters is not None:
            try:
                entrypoint_params = entrypoint.parameters(request_params)
            except ValidationError as e:
                updater.failed(new_agent_text_message(
                    "Refusing request; parameters do not match schema:" + e
                ))
                return
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
