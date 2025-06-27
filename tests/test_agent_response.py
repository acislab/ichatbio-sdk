import base64

import pytest
import pytest_asyncio
from a2a.server.agent_execution import SimpleRequestContextBuilder
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, Part, TextPart, TaskStatusUpdateEvent, FilePart, FileWithUri, FileWithBytes

from ichatbio.agent_response import ResponseChannel, DirectResponse, ProcessBeginResponse, ProcessLogResponse, \
    ArtifactResponse

CONTEXT_ID = "1234"
TASK_ID = "abcd"


@pytest_asyncio.fixture
async def event_queue():
    yield EventQueue()


@pytest.fixture
def message_queue(event_queue):
    yield event_queue.queue


@pytest_asyncio.fixture
async def channel(event_queue):
    request_context = await SimpleRequestContextBuilder().build(context_id=CONTEXT_ID, task_id=TASK_ID)
    task_updater = TaskUpdater(event_queue, request_context.task_id, request_context.context_id)
    yield ResponseChannel(request_context, task_updater)


@pytest.mark.asyncio
async def test_submit_direct_response(channel, message_queue):
    await channel.submit(DirectResponse("hello"), context_id=CONTEXT_ID)
    assert message_queue.qsize() == 1

    message: TaskStatusUpdateEvent = await message_queue.get()
    metadata = {"ichatbio_type": "direct_response", "ichatbio_context_id": CONTEXT_ID}
    assert message.status.message.parts == [
        Part(root=TextPart(text="hello", metadata=metadata))
    ]


@pytest.mark.asyncio
async def test_submit_direct_response_with_data(channel, message_queue):
    await channel.submit(DirectResponse("hello", data={"field": "value"}), context_id=CONTEXT_ID)
    assert message_queue.qsize() == 1

    message: TaskStatusUpdateEvent = await message_queue.get()
    metadata = {"ichatbio_type": "direct_response", "ichatbio_context_id": CONTEXT_ID}
    assert message.status.message.parts == [
        Part(root=TextPart(text="hello", metadata=metadata)),
        Part(root=DataPart(data={"field": "value"}, metadata=metadata))
    ]


@pytest.mark.asyncio
async def test_submit_begin_process(channel, message_queue):
    await channel.submit(ProcessBeginResponse("thinking"), context_id=CONTEXT_ID)
    assert message_queue.qsize() == 1

    message: TaskStatusUpdateEvent = await message_queue.get()
    metadata = {"ichatbio_type": "begin_process_response", "ichatbio_context_id": CONTEXT_ID}
    assert message.status.message.parts == [
        Part(root=TextPart(text="thinking", metadata=metadata))
    ]


@pytest.mark.asyncio
async def test_submit_begin_process_with_data(channel, message_queue):
    await channel.submit(ProcessBeginResponse("thinking", data={"field": "value"}), context_id=CONTEXT_ID)
    assert message_queue.qsize() == 1

    message: TaskStatusUpdateEvent = await message_queue.get()
    metadata = {"ichatbio_type": "begin_process_response", "ichatbio_context_id": CONTEXT_ID}
    assert message.status.message.parts == [
        Part(root=TextPart(text="thinking", metadata=metadata)),
        Part(root=DataPart(data={"field": "value"}, metadata=metadata))
    ]


@pytest.mark.asyncio
async def test_submit_process_log(channel, message_queue):
    await channel.submit(ProcessLogResponse("doing stuff"), context_id=CONTEXT_ID)
    assert message_queue.qsize() == 1

    message: TaskStatusUpdateEvent = await message_queue.get()
    metadata = {"ichatbio_type": "process_log_response", "ichatbio_context_id": CONTEXT_ID}
    assert message.status.message.parts == [
        Part(root=TextPart(text="doing stuff", metadata=metadata))
    ]


@pytest.mark.asyncio
async def test_submit_process_log_with_data(channel, message_queue):
    await channel.submit(ProcessLogResponse("doing stuff", data={"field": "value"}), context_id=CONTEXT_ID)
    assert message_queue.qsize() == 1

    message: TaskStatusUpdateEvent = await message_queue.get()
    metadata = {"ichatbio_type": "process_log_response", "ichatbio_context_id": CONTEXT_ID}
    assert message.status.message.parts == [
        Part(root=TextPart(text="doing stuff", metadata=metadata)),
        Part(root=DataPart(data={"field": "value"}, metadata=metadata))
    ]


@pytest.mark.asyncio
async def test_submit_artifact_with_online_content(channel, message_queue):
    await channel.submit(
        ArtifactResponse(
            mimetype="text/plain",
            description="test artifact",
            uris=["https://test.artifact"],
            metadata={"source": "nowhere"}
        ),
        context_id=CONTEXT_ID
    )
    assert message_queue.qsize() == 1

    message: TaskStatusUpdateEvent = await message_queue.get()
    metadata = {"ichatbio_type": "artifact_response", "ichatbio_context_id": CONTEXT_ID}
    assert message.status.message.parts == [
        Part(
            root=FilePart(
                file=FileWithUri(mimeType="text/plain", name="test artifact", uri="https://test.artifact"),
                metadata=metadata
            )
        ),
        Part(
            root=DataPart(
                data={"uris": ["https://test.artifact"], "metadata": {"source": "nowhere"}},
                metadata=metadata
            )
        )
    ]


@pytest.mark.asyncio
async def test_submit_artifact_with_offline_content(channel, message_queue):
    await channel.submit(
        ArtifactResponse(
            mimetype="text/plain",
            description="test artifact",
            content=b"hello",
            metadata={"source": "nowhere"}
        ),
        context_id=CONTEXT_ID
    )
    assert message_queue.qsize() == 1

    message: TaskStatusUpdateEvent = await message_queue.get()
    metadata = {"ichatbio_type": "artifact_response", "ichatbio_context_id": CONTEXT_ID}
    hello = base64.b64encode(b"hello")
    assert message.status.message.parts == [
        Part(
            root=FilePart(
                file=FileWithBytes(mimeType="text/plain", name="test artifact", bytes=hello),
                metadata=metadata)
        ),
        Part(
            root=DataPart(
                data={"uris": [], "metadata": {"source": "nowhere"}},
                metadata=metadata
            )
        )
    ]
