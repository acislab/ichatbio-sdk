import asyncio
import base64
import logging
import traceback
from dataclasses import dataclass
from typing import Optional, override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
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
    DirectResponse, )


@dataclass
class _Request:
    text: str
    data: dict


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
    reason: str

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
        request = None
        try:
            if context.message is None:
                raise BadRequest("Request does not contain a message")

            try:
                # TODO: for now, assume messages begin with a text part and a data part
                request = _Request(
                    context.message.parts[0].root.text,
                    context.message.parts[1].root.data,
                )
            except (IndexError, AttributeError) as e:
                raise BadRequest("Request is not formatted as expected") from e

            try:
                raw_entrypoint_data = request.data["entrypoint"]
                entrypoint_id = raw_entrypoint_data["id"]
                raw_entrypoint_params = (
                    raw_entrypoint_data["parameters"]
                    if "parameters" in raw_entrypoint_data
                    else {}
                )

            except (AttributeError, IndexError, KeyError) as e:
                raise BadRequest("Failed to parse request data") from e

            entrypoint = next(
                (
                    e
                    for e in self.agent.get_agent_card().entrypoints
                    if e.id == entrypoint_id
                ),
                None,
            )

            if not entrypoint:
                raise BadRequest(f'Invalid entrypoint "{entrypoint_id}"')

            if entrypoint.parameters is not None:
                try:
                    entrypoint_params = entrypoint.parameters(**raw_entrypoint_params)
                except ValidationError as e:
                    raise BadRequest(
                        f'Invalid arguments for entrypoint "{entrypoint_id}": {raw_entrypoint_params}'
                    ) from e
            else:
                entrypoint_params = None

            await updater.start_work()

            response_channel = ResponseChannel(context.task_id)

            # Pass the request to the agent
            logging.info(f"Accepting request {request}")
            agent_task = asyncio.create_task(
                self._run_agent(
                    response_channel,
                    request.text,
                    entrypoint_id,
                    entrypoint_params,
                )
            )

            while True:
                icb_message = await response_channel.message_box.get()

                match icb_message:
                    case AgentFinished():
                        await updater.complete()
                        response_channel.message_box.task_done()
                        break

                    case AgentCrashed(reason=reason):
                        await updater.failed(new_agent_text_message(reason,context.context_id,context.task_id))
                        response_channel.message_box.task_done()
                        break

                    case (
                        DirectResponse(kind=kind, text=text, data=data)
                        | ProcessBeginResponse(kind=kind, summary=text, data=data)
                        | ProcessLogResponse(kind=kind, text=text, data=data)
                    ):
                        metadata = {
                            "ichatbio_type": kind,
                            "ichatbio_context_id": context.context_id,
                        }

                        parts = (
                            [
                                TextPart(text=text, metadata=metadata),
                                DataPart(data=data, metadata=metadata),
                            ]
                            if data
                            else [TextPart(text=text, metadata=metadata)]
                        )

                        await updater.start_work(
                            new_agent_parts_message(
                                [Part(root=p) for p in parts],
                                context.context_id,
                                context.task_id,
                            )
                        )
                        response_channel.message_box.task_done()

                    case ArtifactResponse(
                        mimetype=mimetype,
                        description=description,
                        uris=uris,
                        content=content,
                        metadata=artifact_metadata,
                    ):
                        metadata = {
                            "ichatbio_type": "artifact_response",
                            "ichatbio_context_id": context.context_id,
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
                            file = FileWithUri(
                                uri=uris[0], mimeType=mimetype, name=description
                            )
                        else:
                            raise ValueError(
                                "Artifact message must have content or at least one URI"
                            )

                        parts = [
                            FilePart(file=file, metadata=metadata),
                            DataPart(data=data, metadata=metadata),
                        ]

                        await updater.start_work(
                            new_agent_parts_message(
                                [Part(root=p) for p in parts],
                                context.context_id,
                                context.task_id,
                            )
                        )
                        response_channel.message_box.task_done()
                        # TODO: Keep the A2A task and async task alive and wait for an artifact ack

                    case _:
                        agent_task.cancel()
                        response_channel.message_box.task_done()
                        raise ValueError("Bad message type")


        # If something is wrong with the request, mark the task as "rejected"
        except BadRequest as e:
            logging.warning(f"Rejecting incoming request: {request}", exc_info=e)
            message = updater.new_agent_message(
                [
                    Part(
                        root=TextPart(
                            text=f"Request rejected. Reason: {traceback.format_exc(limit=0)}"
                        )
                    )
                ]
            )
            await updater.reject(message)

    async def _run_agent(
        self,
        response_channel: ResponseChannel,
        request: str,
        entrypoint_id: str,
        entrypoint_params,
    ):
        try:
            response_context = ResponseContext(response_channel)
            await self.agent.run(
                response_context, request, entrypoint_id, entrypoint_params
            )
            await response_channel.message_box.put(AgentFinished())

        # If the agent failed to process the request, mark the task as "failed"
        except Exception as e:
            logging.error(
                f"An exception was raised while handling request {request}", exc_info=e
            )
            await response_channel.message_box.put(AgentCrashed(reason=traceback.format_exc(limit=0)))


    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
