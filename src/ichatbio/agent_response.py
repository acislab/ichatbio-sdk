import asyncio
import logging
from contextlib import asynccontextmanager, AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Optional, Literal, Type, Any
from uuid import uuid4

from pydantic import BaseModel, ValidationError
from typing_extensions import TypeVar

from ichatbio.types import Artifact


@dataclass
class DirectResponse:
    text: str
    data: Optional[dict] = None
    response_model: Optional[Type[BaseModel]] = None
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


@dataclass
class ArtifactAck:
    artifact: Artifact


@dataclass
class NoResponse:
    explanation: str


@dataclass
class ModelResponse:
    explanation: str
    value: Any


@dataclass
class DirectResponseAck:
    explanation: str
    value: Any
    complete: bool


class ResponseChannel:
    def __init__(self):
        self._message_box: asyncio.Queue[ResponseMessage] = asyncio.Queue(maxsize=1)

    async def submit(self, message: ResponseMessage):
        await self._message_box.put(message)
        await self._message_box.join()


    @asynccontextmanager
    async def receive(self):
        message = await self._message_box.get()
        yield message
        self._message_box.task_done()


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
    ) -> Artifact:
        """
        Sends a request to iChatBio to register a new artifact. If content is not included, a resolvable URI must be
        specified. If no resolvable URIs are provided, iChatBio will store the content locally and use its SHA-256 hash
        as its identifier.

        Waits for iChatBio to acknowledge the new artifact.

        :param mimetype: The MIME type of the artifact, e.g. ``text/plain``, ``application/json``, ``image/png``.
        :param description: A brief description of the artifact. Descriptions over ~50 characters may be abbreviated.
        :param uris: Unique identifiers for the artifact. If URIs are resolvable, content can be omitted.
        :param content: The raw content of the artifact.
        :param metadata: Anything related to the artifact, e.g. provenance, schema, landing page URLs, related artifact URIs.
        :returns: Artifact metadata received by iChatBio, which may include new URIs to identify the artifact and locate its content.
        """
        await self._submit_if_active(
            ArtifactResponse(mimetype, description, uris, content, metadata)
        )

        async with self._channel.receive() as m:
            message = m

        match message:
            case ArtifactAck(artifact=artifact):
                return artifact
            case _:
                raise ValueError("Received unexpected message type")


TModel = TypeVar("TModel", bound=BaseModel)

class ResponseContext:
    """
    Provides methods for responding to requests and initiating processes.
    """

    def __init__(self, channel: ResponseChannel):
        self._channel = channel

    async def reply(self, text: Optional[str], data: Optional[dict] = None, response_model: Type[TModel] = None) -> ModelResponse | NoResponse:
        """
        Responds directly to iChatBio, not the user. Text messages can be used to:
        - Request more information
        - Refuse iChatBio's request
        - Provide context for process and artifact messages
        - Provide advice on what to do next
        - etc.

        :param text: A natural language response to iChatBio's request.
        :param data: Structured information related to the message.
        :param response_model: If provided, iChatBio will send a response that tries to follow this model.
        :raises ValidationError: If ``response_model`` is provided and iChatBio's response fails validation.
        :returns: If ``response_model`` is provided, iChatBio's response is returned. Otherwise, returns ``None``.
        """
        logging.info(f'Sending reply "{text}" with data {data}')
        await self._channel.submit(DirectResponse(text, data, response_model))

        if response_model is None:
            return

        logging.info("Waiting for iChatBio's response")
        async with self._channel.receive() as m:
            message = m

        match message:
            case DirectResponseAck(explanation=explanation, complete=False):
                return NoResponse(explanation)
            case DirectResponseAck(explanation=explanation, value=unvalidated_response, complete=True):
                try:
                    return ModelResponse(explanation, response_model.model_validate(unvalidated_response))
                except ValidationError as e:
                    logging.warning("Response from iChatBio does not satisfy model constraints", e)
                    raise
            case _:
                raise ValueError("Received unexpected message type")


    @asynccontextmanager
    async def begin_process(
        self, summary: str, metadata: Optional[dict] = None
    ) -> AbstractAsyncContextManager[IChatBioAgentProcess]:
        """
        Begins a long-running process to log agent actions and create artifacts as outputs. Users of iChatBio will see a visual representation of the process with the provided summary, and be able to inspect the process to review all recorded log messages and artifacts.

        :param summary: A brief summary of what the agent is doing, e.g. "Searching iDigBio".
        :param metadata: Optional structured information to contextualize the process.
        :return:

        Processes should be started using an ``async with`` statement::

            async with context.begin_process("Searching iDigBio") as process:
                # Some IDEs don't infer what ``process`` is, so provide a hint:
                process: IChatBioAgentProcess

                # Agent actions, e.g.:
                await process.log("Generating search parameters")
                await process.create_artifact()

        """
        process = IChatBioAgentProcess(self._channel, summary, metadata)
        await process._begin()
        try:
            yield process
        finally:
            await process._end()
