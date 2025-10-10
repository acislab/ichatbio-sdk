import asyncio
import logging
from contextlib import asynccontextmanager, AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Optional, Literal
from uuid import uuid4


@dataclass
class DirectResponse:
    text: str
    data: Optional[dict] = None
    kind: Literal["direct_response"] = "direct_response"


@dataclass
class ProcessBeginResponse:
    summary: str
    data: Optional[dict] = None
    kind: Literal["begin_process_response"] = "begin_process_response"


@dataclass
class ProcessLogResponse:
    text: str
    data: Optional[dict] = None
    kind: Literal["process_log_response"] = "process_log_response"


@dataclass
class ArtifactResponse:
    mimetype: str
    description: str
    uris: Optional[list[str]] = None
    content: Optional[bytes] = None
    metadata: Optional[dict] = None
    kind: Literal["artifact_response"] = "artifact_response"


ResponseMessage = (
    DirectResponse | ProcessBeginResponse | ProcessLogResponse | ArtifactResponse
)


class ResponseChannel:
    def __init__(self, task_id: str):
        self.message_box: asyncio.Queue[ResponseMessage] = asyncio.Queue(maxsize=1)
        self.task_id = task_id

    async def submit(self, message: ResponseMessage):
        await self.message_box.put(message)
        await self.message_box.join()


class IChatBioAgentProcess:
    def __init__(
        self, channel: ResponseChannel, summary: str, metadata: Optional[dict]
    ):
        self._channel = channel
        self._summary = summary
        self._metadata = metadata
        self._context_id = None

    async def _submit_if_active(self, message: ResponseMessage):
        if not self._context_id:
            raise ValueError("Process is not yet started")
        if not self._channel:
            raise ValueError("Process is over")
        await self._channel.submit(message)

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
        await self._submit_if_active(
            ProcessBeginResponse(self._summary, self._metadata)
        )

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
        metadata: Optional[dict] = None,
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
        await self._submit_if_active(
            ArtifactResponse(mimetype, description, uris, content, metadata)
        )


class ResponseContext:
    """
    Provides methods for responding to requests and initiating processes.
    """

    def __init__(self, channel: ResponseChannel):
        self._channel = channel
        self._root_context_id = channel.task_id

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
        logging.info(f'Sending reply "{text}" with data {data}')
        await self._channel.submit(DirectResponse(text, data))

    @asynccontextmanager
    async def begin_process(
        self, summary: str, metadata: Optional[dict] = None
    ) -> AbstractAsyncContextManager[IChatBioAgentProcess]:
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
