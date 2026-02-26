import asyncio
import base64
import logging
import traceback
from dataclasses import dataclass
from typing import Optional, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Message
from a2a.types import Part, FilePart, DataPart, FileWithUri, FileWithBytes
from a2a.types import UnsupportedOperationError, TextPart
from a2a.utils import new_agent_parts_message, new_agent_text_message
from a2a.utils.errors import ServerError
from pydantic import ValidationError

from ichatbio.agent import IChatBioAgent
from ichatbio.agent_response import (
    ResponseContext,
    ResponseChannel,
    ArtifactResponse,
    ProcessLogResponse,
    ProcessBeginResponse,
    DirectResponse,
)


class BadRequest(ValueError):
    pass


def _make_text_and_data_parts(text: str, data: Optional[dict], metadata: dict):
    yield TextPart(text=text, metadata=metadata)
    if data is not None:
        yield DataPart(data=data, metadata=metadata)


@dataclass
class AgentFinished:
    pass


@dataclass
class AgentCrashed:
    exc: Exception


@dataclass
class Request:
    text: str
    entrypoint: str
    arguments: Optional[dict]


def new_agent_response_message(
    parts: list[TextPart | FilePart | DataPart], context_id: str, task_id: str
):
    return new_agent_parts_message(
        [Part(root=p) for p in parts],
        context_id,
        task_id,
    )


def make_artifact_parts(
    artifact_metadata: dict,
    content: bytes,
    description: str,
    mimetype: str,
    uris: list[str],
) -> list[TextPart | FilePart | DataPart]:
    metadata = {
        "ichatbio": {
            "message_type": "artifact_response"
        },
        "ichatbio_type": "artifact_response" # TODO: remove with version 3
    }

    data = {
        "metadata": artifact_metadata,
        "uris": uris if uris else [],
    }

    if content is not None:
        file = FileWithBytes(
            bytes=base64.b64encode(content),
            mimeType=mimetype,
            name=description,
        )
    elif uris:
        file = FileWithUri(uri=uris[0], mimeType=mimetype, name=description)
    else:
        raise ValueError("Artifact message must have content or at least one URI")

    return [
        FilePart(file=file, metadata=metadata),
        DataPart(data=data, metadata=metadata),
    ]


def make_text_data_parts(kind: str, text: str, data: dict):
    metadata = {
        "ichatbio": {
            "message_type": kind
        },
        "ichatbio_type": kind # TODO: remove with version 3
    }

    if data:
        return [
            TextPart(text=text, metadata=metadata),
            DataPart(data=data, metadata=metadata),
        ]
    else:
        return [TextPart(text=text, metadata=metadata)]


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

        if context.current_task:
            return

        # Start a new task
        await updater.submit()

        # Process the request
        try:
            request = await self.parse_request(context.message)
        # If something is wrong with the request, mark the task as "rejected"
        except BadRequest as e:
            logging.warning(f"Rejecting request: {context.message}", exc_info=e)
            await updater.reject(
                updater.new_agent_message(
                    [
                        Part(
                            root=TextPart(
                                text=f"Request rejected. Reason: {traceback.format_exc(limit=0)}"
                            )
                        )
                    ]
                )
            )
            return

        logging.info(f"Accepting request: {request}")
        await updater.start_work()

        # Run the agent in a separate task to produce response messages
        response_channel = ResponseChannel(context.task_id)
        agent_task = asyncio.create_task(self.run_agent(response_channel, request))

        # Consume agent response messages
        while True:
            icb_message = await response_channel.message_box.get()

            try:
                match icb_message:
                    case AgentFinished():
                        await updater.complete()
                        break

                    case AgentCrashed(exc=exc):
                        await updater.failed(
                            new_agent_text_message(
                                "".join(traceback.format_exception(exc, limit=0)),
                                context.context_id,
                                context.task_id,
                            )
                        )
                        raise exc

                    case (
                        DirectResponse(kind=kind, text=text, data=data)
                        | ProcessBeginResponse(kind=kind, summary=text, data=data)
                        | ProcessLogResponse(kind=kind, text=text, data=data)
                    ):
                        await updater.start_work(
                            new_agent_response_message(
                                make_text_data_parts(kind, text, data),
                                context.context_id,
                                context.task_id,
                            )
                        )

                    case ArtifactResponse(
                        mimetype=mimetype,
                        description=description,
                        uris=uris,
                        content=content,
                        metadata=artifact_metadata,
                    ):
                        await updater.start_work(
                            new_agent_response_message(
                                make_artifact_parts(
                                    artifact_metadata,
                                    content,
                                    description,
                                    mimetype,
                                    uris,
                                ),
                                context.context_id,
                                context.task_id,
                            )
                        )
                        # TODO: Keep the A2A task and async task alive and wait for an artifact ack

                    case _:
                        agent_task.cancel()
                        raise ValueError(
                            f'Unexpected response message type "{type(icb_message)}": {icb_message}'
                        )
            finally:
                response_channel.message_box.task_done()

    async def parse_request(self, request: Message):
        match request:
            # TODO: for now, assume messages begin with a text part and a data part
            case Message(
                parts=[
                    Part(root=TextPart(text=text)),
                    Part(root=DataPart(data=data)),
                ]
            ):
                request_text = text
                request_data = data
            case _:
                raise BadRequest("Request does not contain the expected message parts")

        match request_data:
            case {
                "entrypoint": {"id": entrypoint_id, "parameters": raw_entrypoint_params}
            }:
                entrypoint = self._get_agent_entrypoint(entrypoint_id)
                if not entrypoint.parameters:
                    raise BadRequest("Entrypoint does not take arguments")

                try:
                    entrypoint_params = entrypoint.parameters(**raw_entrypoint_params)
                except ValidationError as e:
                    raise BadRequest("Invalid entrypoint arguments") from e

            case {"entrypoint": {"id": entrypoint_id}}:
                entrypoint = self._get_agent_entrypoint(entrypoint_id)
                if entrypoint.parameters:
                    try:
                        entrypoint_params = entrypoint.parameters()
                    except ValidationError:
                        raise BadRequest("Missing entrypoint arguments")
                else:
                    entrypoint_params = None

            case _:
                raise BadRequest("Failed to parse entrypoint data")

        return Request(request_text, entrypoint_id, entrypoint_params)

    def _get_agent_entrypoint(self, entrypoint_id):
        entrypoint = next(
            (
                e
                for e in self.agent.get_agent_card().entrypoints
                if e.id == entrypoint_id
            ),
            None,
        )

        if not entrypoint:
            raise BadRequest(f"Unrecognized entrypoint")

        return entrypoint

    async def run_agent(
        self,
        response_channel: ResponseChannel,
        request: Request,
    ):
        try:
            response_context = ResponseContext(response_channel)
            await self.agent.run(
                response_context, request.text, request.entrypoint, request.arguments
            )
            await response_channel.message_box.put(AgentFinished())

        # If the agent failed to process the request, mark the task as "failed"
        except Exception as e:
            logging.error(
                f"An exception was raised while handling request {request}", exc_info=e
            )
            await response_channel.message_box.put(AgentCrashed(e))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
