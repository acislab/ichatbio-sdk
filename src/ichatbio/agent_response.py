import base64
import logging
from contextlib import asynccontextmanager, AbstractAsyncContextManager
from typing import Optional
from uuid import uuid4

from a2a.server.agent_execution import RequestContext
from a2a.server.tasks import TaskUpdater
from a2a.types import TextPart, FilePart, DataPart, TaskState, Part, FileWithBytes, FileWithUri
from a2a.utils import new_agent_parts_message
from attr import dataclass


@dataclass
class DirectResponse:
    text: str
    data: Optional[dict] = None


@dataclass
class ProcessBeginResponse:
    summary: str
    data: Optional[dict] = None


@dataclass
class ProcessLogResponse:
    text: str
    data: Optional[dict] = None


@dataclass
class ArtifactResponse:
    mimetype: str
    description: str
    uris: Optional[list[str]] = None
    content: Optional[bytes] = None
    metadata: Optional[dict] = None


ResponseMessage = DirectResponse | ProcessBeginResponse | ProcessLogResponse | ArtifactResponse


class ResponseChannel:
    def __init__(self, context: RequestContext, updater: TaskUpdater):
        self._context = context
        self._updater = updater

    async def submit(self, message: ResponseMessage, context_id: str):
        match message:
            case DirectResponse(text=text, data=data):
                metadata = {
                    "ichatbio_type": "direct_response",
                    "ichatbio_context_id": context_id
                }

                parts = [TextPart(text=text, metadata=metadata)]
                if data is not None:
                    parts.append(DataPart(data=data, metadata=metadata))

            case ProcessBeginResponse(summary=summary, data=data):
                metadata = {
                    "ichatbio_type": "begin_process_response",
                    "ichatbio_context_id": context_id
                }

                parts = [TextPart(text=summary, metadata=metadata)]
                if data is not None:
                    parts.append(DataPart(data=data, metadata=metadata))

            case ProcessLogResponse(text=text, data=data):
                metadata = {
                    "ichatbio_type": "process_log_response",
                    "ichatbio_context_id": context_id
                }

                parts = [TextPart(text=text, metadata=metadata)]
                if data is not None:
                    parts.append(DataPart(data=data, metadata=metadata))

            case ArtifactResponse(mimetype=mimetype, description=description, uris=uris, content=content,
                                  metadata=artifact_metadata):
                metadata = {
                    "ichatbio_type": "artifact_response",
                    "ichatbio_context_id": context_id
                }
                data = {
                    "metadata": artifact_metadata,
                    "uris": uris if uris else []
                }

                if content is not None:
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
                    raise ValueError("Artifact message must have content or at least one URI")

                parts = [
                    FilePart(file=file, metadata=metadata),
                    DataPart(data=data, metadata=metadata)
                ]

            case _:
                raise ValueError("Bad message type")

        await self._updater.update_status(
            TaskState.working,
            new_agent_parts_message(
                [Part(root=p) for p in parts],
                self._context.context_id,
                self._context.task_id)
        )


class IChatBioAgentProcess:
    def __init__(self, channel: ResponseChannel, summary: str, metadata: Optional[dict]):
        self._channel = channel
        self._summary = summary
        self._metadata = metadata
        self._context_id = None

    async def _submit_if_active(self, message: ResponseMessage):
        if not self._context_id:
            raise ValueError("Process is not yet started")
        if not self._channel:
            raise ValueError("Process is over")
        await self._channel.submit(message, self._context_id)

    async def _begin(self):
        """
        Do not call this function directly. It will be performed automatically when beginning the process in a "with"
        statement.

        >>> with context.begin_process(...) as process:
        >>>     # process._begin() is called immediately
        """
        if self._context_id:
            raise ValueError("Process has already started")
        self._context_id = str(uuid4())
        await self._submit_if_active(ProcessBeginResponse(self._summary, self._metadata))

    async def _end(self):
        """
        Do not call this function directly. It will be performed automatically when beginning the process in a "with"
        statement.

        >>> with context.begin_process(...) as process:
        >>>     # process._end() is called at the end of this block
        """
        if not self._context_id:
            raise ValueError("Process is not yet started")
        if not self._channel:
            raise ValueError("Process is already over")
        self._channel = None

    async def log(self, text: str, data: dict = None):
        """
        Logs the agent's actions and outcomes. iChatBio users will see these messages in Markdown formatting.
        """
        await self._submit_if_active(ProcessLogResponse(text, data))

    async def create_artifact(
            self,
            mimetype: str,
            description: str,
            uris: Optional[list[str]] = None,
            content: Optional[bytes] = None,
            metadata: Optional[dict] = None
    ):
        """
        Returns an identifiable digital object to iChatBio. If content is not included, a resolvable URI must be
        specified. If no resolvable URIs are provided, iChatBio will store the content locally and use its SHA-256 hash
        as its identifier.

        :param mimetype: The MIME type of the artifact, e.g. ``text/plain``, ``application/json``, ``image/png``.
        :param description: A brief description of the artifact. Descriptions over ~50 characters may be abbreviated.
        :param uris: Unique identifiers for the artifact. If URIs are resolvable, content can be omitted.
        :param content: The raw content of the artifact.
        :param metadata: Anything related to the artifact, e.g. provenance, schema, landing page URLs, related artifact URIs.
        """
        await self._submit_if_active(ArtifactResponse(mimetype, description, uris, content, metadata))


class ResponseContext:
    """
    Provides methods for responding to requests and initiating processes.
    """

    def __init__(self, channel: ResponseChannel, root_context_id: str):
        self._channel = channel
        self._root_context_id = root_context_id

    async def reply(self, text: Optional[str], data: Optional[dict] = None):
        """
        Responds directly to the iChatBio assistant, not the user. Text messages can be used to:
        - Request more information
        - Refuse the assistant's request
        - Provide context for process and artifact messages
        - Provide advice on what to do next
        - etc.

        Open a process to instead respond with process logs or persistent artifacts.
        :param text: A natural language response to the assistant's request.
        :param data: Structured information related to the message.
        """
        logging.info(f"Sending reply \"{text}\" with data {data}")
        await self._channel.submit(DirectResponse(text, data), self._root_context_id)

    @asynccontextmanager
    async def begin_process(self, summary: str, metadata: Optional[dict] = None) -> \
            AbstractAsyncContextManager[IChatBioAgentProcess]:
        """
        Begins a long-running process to log agent actions and create artifacts as outputs. Users of iChatBio will see a visual representation of the process with the provided summary, and be able to inspect the process to review all recorded log messages and artifacts.

        :param summary: A brief summary of what the agent is doing, e.g. "Searching iDigBio".
        :param metadata: Optional structured information to contextualize the process.
        :return:

        Processes must be started using a ``with`` statement:

            with context.process("Searching iDigBio") as process:
                process.log("Generating search parameters")
                # Agent actions
                process.create_artifact()

        """
        process = IChatBioAgentProcess(self._channel, summary, metadata)
        await process._begin()
        try:
            yield process
        finally:
            await process._end()
