import asyncio
import base64
import importlib.metadata
from typing import Iterable, override

import a2a.server.events
import pytest
import pytest_asyncio
from a2a.server.agent_execution import SimpleRequestContextBuilder
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
    Task,
)
from pydantic import BaseModel

import ichatbio.types
from ichatbio.agent_executor import (
    IChatBioAgentExecutor,
    AgentCrashed,
    AgentFinished,
    Request,
    SuspendedTask,
)
from ichatbio.agent_response import (
    ResponseChannel,
    DirectResponse,
    ProcessBeginResponse,
    ProcessLogResponse,
    ArtifactResponse,
    DirectResponseAck,
)
from ichatbio.agent_response import ResponseMessage


class MockExecutor(IChatBioAgentExecutor):
    _agent_messages: Iterable[ResponseMessage]

    def __init__(self):
        super().__init__(None)

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
            await response_channel.submit(AgentFinished())
        except asyncio.CancelledError:
            raise
        except Exception as e:
            await response_channel.submit(AgentCrashed(e))


def read_events(event_queue):
    message_ids = (f"message-{i}" for i in range(1, 100))
    while True:
        event = event_queue.queue.get_nowait()
        event.status.timestamp = None
        if event.status.message:
            event.status.message.message_id = next(message_ids)

        yield event

        match event:
            case TaskStatusUpdateEvent(final=True):
                break


@pytest_asyncio.fixture
async def event_queue():
    yield EventQueue()


@pytest.fixture
def message_queue(event_queue):
    yield event_queue.queue


@pytest_asyncio.fixture
async def channel(event_queue):
    yield ResponseChannel()


@pytest.fixture
def executor():
    return MockExecutor()


@pytest_asyncio.fixture
async def execute(event_queue, executor):
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

    async def run(
        *agent_messages: ResponseMessage,
        request: a2a.types.Message = None,
        task: Task = None,
    ) -> list[a2a.server.events.Event]:
        if request is None:
            request = a2a.types.Message(
                parts=[Part(root=part) for part in parts],
                role=a2a.types.Role.user,
                message_id="request-1",
            )

        request_context = await SimpleRequestContextBuilder().build(
            context_id="context-1",
            task_id="task-1",
            task=task,
            params=MessageSendParams(message=request),
        )

        executor.agent_response(*agent_messages)
        await executor.execute(request_context, event_queue)
        return list(read_events(event_queue))

    return run


@pytest.mark.asyncio
async def test_executor(execute):
    events = await execute(DirectResponse("hello"))

    assert events == [
        Task(
            context_id="context-1",
            id="task-1",
            status=TaskStatus(state=TaskState.submitted),
            history=[
                Message(
                    context_id="context-1",
                    message_id="request-1",
                    task_id="task-1",
                    role="user",
                    parts=[
                        Part(root=TextPart(text="request text")),
                        Part(
                            root=DataPart(
                                data={
                                    "entrypoint": {
                                        "id": "fake_entrypoint",
                                        "parameters": {
                                            "fake_parameter": "doesn't matter"
                                        },
                                    }
                                }
                            )
                        ),
                    ],
                )
            ],
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
                    task_id="task-1",
                    role=Role.agent,
                    metadata={
                        "ichatbio": {
                            "sdk": importlib.metadata.version("ichatbio-sdk"),
                            "message_type": "direct_response",
                        }
                    },
                    parts=[Part(root=TextPart(text="hello"))],
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
async def test_ask_question(execute):
    class ResponseModel(BaseModel):
        answer: int

    events = await execute(
        DirectResponse(text="What's the answer?", response_model=ResponseModel)
    )

    message = events[2].status.message
    assert message.metadata["ichatbio"]["response_model"] == {
        "properties": {"answer": {"title": "Answer", "type": "integer"}},
        "required": ["answer"],
        "title": "ResponseModel",
        "type": "object",
    }
    assert message.parts == [
        Part(root=TextPart(metadata=None, text="What's the answer?"))
    ]

    assert events[-1].status.state == TaskState.input_required


@pytest.mark.asyncio
async def test_answer_question(execute, executor):
    class ResponseModel(BaseModel):
        answer: int

    channel = ResponseChannel()
    response_box = [None]

    async def work():
        nonlocal response_box
        async with channel.receive() as response:
            response_box = [response]
        await channel.submit(AgentFinished())

    agent_task = asyncio.create_task(work())

    executor.suspended_tasks["task-1"] = SuspendedTask(agent_task, channel)

    assert response_box[0] is None

    events = await execute(
        DirectResponse("Got it!"),
        request=Message(
            message_id="message-2",
            task_id="task-1",
            role="user",
            parts=[
                DataPart(
                    data={
                        "type": "query_response",
                        "explanation": "I ran the numbers",
                        "model_response": ResponseModel(answer=99).model_dump(),
                        "complete": True,
                    }
                ),
            ],
        ),
        task=Task(
            context_id="context-1",
            id="task-1",
            status=TaskStatus(state=TaskState.input_required),
        ),
    )

    assert response_box[0] == DirectResponseAck(
        explanation="I ran the numbers", value={"answer": 99}, complete=True
    )
    assert events[-1].status.state == TaskState.completed


@pytest.mark.asyncio
async def test_refuse_to_answer(execute, executor):
    channel = ResponseChannel()
    response_box = [None]

    async def work():
        nonlocal response_box
        async with channel.receive() as response:
            response_box = [response]
        await channel.submit(AgentFinished())

    agent_task = asyncio.create_task(work())

    executor.suspended_tasks["task-1"] = SuspendedTask(agent_task, channel)

    assert response_box[0] is None

    events = await execute(
        DirectResponse("Got it!"),
        request=Message(
            message_id="message-2",
            task_id="task-1",
            role="user",
            parts=[
                DataPart(
                    data={
                        "type": "query_response",
                        "explanation": "I give up :(",
                        "model_response": None,
                        "complete": False,
                    }
                ),
            ],
        ),
        task=Task(
            context_id="context-1",
            id="task-1",
            status=TaskStatus(state=TaskState.input_required),
        ),
    )

    assert response_box[0] == DirectResponseAck(
        explanation="I give up :(", value=None, complete=False
    )
    assert events[-1].status.state == TaskState.completed


@pytest.mark.asyncio
async def test_submit_direct_response_with_data(execute):
    events = await execute(DirectResponse("hello", data={"name": "barb"}))

    message = events[2].status.message
    assert message.metadata["ichatbio"]["message_type"] == "direct_response"
    assert message.parts == [
        Part(root=TextPart(text="hello")),
        Part(root=DataPart(data={"name": "barb"})),
    ]


@pytest.mark.asyncio
async def test_submit_begin_process(execute):
    events = await execute(ProcessBeginResponse("thinking"))

    message = events[2].status.message
    assert message.metadata["ichatbio"]["message_type"] == "begin_process_response"
    assert message.parts == [Part(root=TextPart(text="thinking"))]


@pytest.mark.asyncio
async def test_submit_process_log(execute):
    events = await execute(ProcessLogResponse("doing stuff"))

    message = events[2].status.message
    assert message.metadata["ichatbio"]["message_type"] == "process_log_response"
    assert message.parts == [Part(root=TextPart(kind="text", text="doing stuff"))]


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

    message = events[2].status.message
    assert message.metadata["ichatbio"]["message_type"] == "artifact_response"
    assert message.parts == [
        Part(
            root=FilePart(
                file=FileWithUri(
                    mimeType="text/plain",
                    name="test artifact",
                    uri="https://test.artifact",
                )
            )
        ),
        Part(
            root=DataPart(
                data={
                    "uris": ["https://test.artifact"],
                    "metadata": {"source": "nowhere"},
                }
            )
        ),
    ]

    assert events[-1].status.state == TaskState.input_required


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

    message = events[2].status.message
    assert message.metadata["ichatbio"]["message_type"] == "artifact_response"
    assert message.parts == [
        Part(
            root=FilePart(
                file=FileWithBytes(
                    mimeType="text/plain",
                    name="test artifact",
                    bytes=base64.b64encode(b"hello"),
                )
            )
        ),
        Part(root=DataPart(data={"metadata": {"source": "nowhere"}, "uris": []})),
    ]

    assert events[-1].status.state == TaskState.input_required


@pytest.mark.asyncio
async def test_receive_artifact_ack(execute, executor):
    channel = ResponseChannel()

    goodie = object()
    goodie_box = [None]

    async def work():
        nonlocal goodie_box
        async with channel.receive():
            goodie_box = [goodie]
        await channel.submit(AgentFinished())

    agent_task = asyncio.create_task(work())

    executor.suspended_tasks["task-1"] = SuspendedTask(agent_task, channel)
    artifact = ichatbio.types.Artifact(
        local_id="#0000",
        mimetype="text/plain",
        description="Test artifact",
        uris=["hash://sha256//blah"],
        metadata={},
    )

    assert goodie_box[0] is None

    events = await execute(
        DirectResponse("Got it!"),
        request=Message(
            message_id="message-2",
            task_id="task-1",
            role="user",
            parts=[
                DataPart(data={"artifact": artifact.model_dump()}),
            ],
        ),
        task=Task(
            context_id="context-1",
            id="task-1",
            status=TaskStatus(state=TaskState.input_required),
        ),
    )

    assert goodie_box[0] == goodie
    assert events[-1].status.state == TaskState.completed
