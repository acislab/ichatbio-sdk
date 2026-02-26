import base64
from typing import Iterable, override

import a2a.server.events
import pytest
import pytest_asyncio
from a2a.server.agent_execution import SimpleRequestContextBuilder, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    DataPart,
    Part,
    TextPart,
    TaskStatusUpdateEvent,
    FilePart,
    FileWithUri,
    FileWithBytes,
    MessageSendParams,
    Message,
    TaskStatus,
    TaskState,
    Role,
)

from ichatbio.agent_executor import (
    IChatBioAgentExecutor,
    AgentCrashed,
    AgentFinished,
    Request,
)
from ichatbio.agent_response import (
    ResponseChannel,
    DirectResponse,
    ProcessBeginResponse,
    ProcessLogResponse,
    ArtifactResponse,
)
from ichatbio.agent_response import ResponseMessage


class MockExecutor(IChatBioAgentExecutor):
    _agent_messages: Iterable[ResponseMessage]

    def __init__(self, request_context: RequestContext, event_queue: EventQueue):
        super().__init__(None)
        self.request_context = request_context
        self.event_queue = event_queue

    def agent_response(self, *messages):
        self._agent_messages = messages

    @override
    async def parse_request(self, request: Message):
        return Request("test request", "test_entrypoint", None)

    @override
    async def run_agent(self, response_channel: ResponseChannel, request: Request):
        try:
            for message in self._agent_messages:
                await response_channel.submit(message)
            await response_channel.message_box.put(AgentFinished())
        except Exception as e:
            await response_channel.message_box.put(AgentCrashed(e))


def read_events(event_queue):
    message_ids = (f"message-{i}" for i in range(1, 100))
    while True:
        event = event_queue.queue.get_nowait()
        event.status.timestamp = None
        if event.status.message:
            event.status.message.message_id = next(message_ids)

        yield event

        if event.final:
            break


@pytest_asyncio.fixture
async def event_queue():
    yield EventQueue()


@pytest.fixture
def message_queue(event_queue):
    yield event_queue.queue


@pytest_asyncio.fixture
async def channel(event_queue):
    yield ResponseChannel("task-1")


@pytest_asyncio.fixture
async def execute(event_queue):
    text = "request text"
    data = {
        "entrypoint": {
            "id": "fake_entrypoint",
            "parameters": {"fake_parameter": "doesn't matter"},
        }
    }

    parts = [
        TextPart(text=text),
        DataPart(data=data),
    ]

    request_context = await SimpleRequestContextBuilder().build(
        context_id="context-1",
        task_id="task-1",
        params=MessageSendParams(
            message=a2a.types.Message(
                parts=[Part(root=part) for part in parts],
                role=a2a.types.Role.user,
                message_id="request-1",
            )
        ),
    )

    executor = MockExecutor(request_context, event_queue)

    async def run(*agent_messages: ResponseMessage) -> list[a2a.server.events.Event]:
        executor.agent_response(*agent_messages)
        await executor.execute(request_context, event_queue)
        return list(read_events(event_queue))

    return run


@pytest.mark.asyncio
async def test_executor(execute):
    events = await execute(DirectResponse("hello"))

    assert events == [
        TaskStatusUpdateEvent(
            context_id="context-1",
            final=False,
            status=TaskStatus(state=TaskState.submitted),
            task_id="task-1",
        ),
        TaskStatusUpdateEvent(
            context_id="context-1",
            final=False,
            status=TaskStatus(state=TaskState.working),
            task_id="task-1",
        ),
        TaskStatusUpdateEvent(
            context_id="context-1",
            final=False,
            status=TaskStatus(
                message=Message(
                    context_id="context-1",
                    message_id="message-1",
                    parts=[
                        Part(
                            root=TextPart(
                                metadata={
                                    "https://github.com/acislab/ichatbio-sdk/a2a/v1": { "message_type": "direct_response" },
                                    "ichatbio_type": "direct_response"
                                },
                                text="hello",
                            )
                        )
                    ],
                    role=Role.agent,
                    task_id="task-1",
                ),
                state=TaskState.working,
            ),
            task_id="task-1",
        ),
        TaskStatusUpdateEvent(
            context_id="context-1",
            final=True,
            status=TaskStatus(state=TaskState.completed),
            task_id="task-1",
        ),
    ]


@pytest.mark.asyncio
async def test_submit_direct_response_with_data(execute):
    events = await execute(DirectResponse("hello", data={"name": "barb"}))

    assert events[2].status.message.parts == [
        Part(
            root=TextPart(
                metadata={
                    "https://github.com/acislab/ichatbio-sdk/a2a/v1": { "message_type": "direct_response" },
                    "ichatbio_type": "direct_response"
                },
                text="hello",
            )
        ),
        Part(
            root=DataPart(
                metadata={
                    "https://github.com/acislab/ichatbio-sdk/a2a/v1": { "message_type": "direct_response" },
                    "ichatbio_type": "direct_response"
                },
                data={"name": "barb"},
            )
        ),
    ]


@pytest.mark.asyncio
async def test_submit_begin_process(execute):
    events = await execute(ProcessBeginResponse("thinking"))

    assert events[2].status.message.parts == [
        Part(
            root=TextPart(
                kind="text",
                metadata={
                    "https://github.com/acislab/ichatbio-sdk/a2a/v1": { "message_type": "begin_process_response" },
                    "ichatbio_type": "begin_process_response"
                },
                text="thinking",
            )
        )
    ]


@pytest.mark.asyncio
async def test_submit_process_log(execute):
    events = await execute(ProcessLogResponse("doing stuff"))

    assert events[2].status.message.parts == [
        Part(
            root=TextPart(
                kind="text",
                metadata={
                    "https://github.com/acislab/ichatbio-sdk/a2a/v1": { "message_type": "process_log_response" },
                    "ichatbio_type": "process_log_response"
                },
                text="doing stuff",
            )
        )
    ]


@pytest.mark.asyncio
async def test_submit_artifact_with_online_content(execute):
    events = await execute(
        ArtifactResponse(
            mimetype="text/plain",
            description="test artifact",
            uris=["https://test.artifact"],
            metadata={"source": "nowhere"},
        )
    )

    assert events[2].status.message.parts == [
        Part(
            root=FilePart(
                file=FileWithUri(
                    mimeType="text/plain",
                    name="test artifact",
                    uri="https://test.artifact",
                ),
                metadata={
                    "https://github.com/acislab/ichatbio-sdk/a2a/v1": { "message_type": "artifact_response" },
                    "ichatbio_type": "artifact_response"
                },
            )
        ),
        Part(
            root=DataPart(
                data={
                    "uris": ["https://test.artifact"],
                    "metadata": {"source": "nowhere"},
                },
                metadata={
                    "https://github.com/acislab/ichatbio-sdk/a2a/v1": { "message_type": "artifact_response" },
                    "ichatbio_type": "artifact_response"
                },
            )
        ),
    ]


@pytest.mark.asyncio
async def test_submit_artifact_with_offline_content(execute):
    events = await execute(
        ArtifactResponse(
            mimetype="text/plain",
            description="test artifact",
            content=b"hello",
            metadata={"source": "nowhere"},
        )
    )

    assert events[2].status.message.parts == [
        Part(
            root=FilePart(
                file=FileWithBytes(
                    mimeType="text/plain",
                    name="test artifact",
                    bytes=base64.b64encode(b"hello"),
                ),
                metadata={
                    "https://github.com/acislab/ichatbio-sdk/a2a/v1": { "message_type": "artifact_response" },
                    "ichatbio_type": "artifact_response"
                },
            )
        ),
        Part(
            root=DataPart(
                data={"metadata": {"source": "nowhere"}, "uris": []},
                metadata={
                    "https://github.com/acislab/ichatbio-sdk/a2a/v1": { "message_type": "artifact_response" },
                    "ichatbio_type": "artifact_response"
                },
            )
        ),
    ]
