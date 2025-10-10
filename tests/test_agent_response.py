import asyncio
from typing import Coroutine

import pytest

from ichatbio.agent_response import (
    ResponseChannel,
    DirectResponse,
    ProcessBeginResponse,
    ProcessLogResponse,
    ArtifactResponse,
    ResponseContext,
    IChatBioAgentProcess,
    ResponseMessage,
    ArtifactAck,
)

CONTEXT_ID = "1234"
TASK_ID = "abcd"


@pytest.fixture
def channel():
    return ResponseChannel()


@pytest.fixture
def context(channel):
    return ResponseContext(channel)


@pytest.fixture
def run(channel):
    async def do_work(work: Coroutine) -> list[ResponseMessage]:
        done = object()

        async def work_and_finish():
            await work
            await channel.submit(done)

        task = asyncio.create_task(work_and_finish())

        messages = []
        while True:
            message = await channel.message_box.get()
            channel.message_box.task_done()

            if message is done:
                break

            if isinstance(message, ArtifactResponse):
                await channel.submit(ArtifactAck(None))

            messages.append(message)

        await task
        assert channel.message_box.empty()

        return messages

    return do_work


@pytest.mark.asyncio
async def test_submit_direct_response(run, context):
    messages = await run(context.reply("hi"))
    assert messages == [DirectResponse(text="hi")]


@pytest.mark.asyncio
async def test_submit_direct_response_with_data(run, context):
    messages = await run(context.reply("hi", data={1: 2}))
    assert messages == [DirectResponse(text="hi", data={1: 2})]


@pytest.mark.asyncio
async def test_submit_begin_process(run, context):
    async def work():
        async with context.begin_process("test") as _:
            pass
        pass

    messages = await run(work())
    assert messages == [ProcessBeginResponse(summary="test")]


@pytest.mark.asyncio
async def test_submit_begin_process_with_data(run, context):
    async def work():
        async with context.begin_process("test", {1: 2}) as _:
            pass

    messages = await run(work())
    assert messages == [ProcessBeginResponse(summary="test", data={1: 2})]


@pytest.mark.asyncio
async def test_submit_process_log(run, context):
    async def work():
        async with context.begin_process("test") as process:
            process: IChatBioAgentProcess
            await process.log("working")

    messages = await run(work())
    assert messages == [
        ProcessBeginResponse(summary="test"),
        ProcessLogResponse(text="working"),
    ]


@pytest.mark.asyncio
async def test_submit_process_log_with_data(run, context):
    async def work():
        async with context.begin_process("test") as process:
            process: IChatBioAgentProcess
            await process.log("working", {1: 2})

    messages = await run(work())
    assert messages == [
        ProcessBeginResponse(summary="test"),
        ProcessLogResponse(text="working", data={1: 2}),
    ]


@pytest.mark.asyncio
async def test_submit_artifact(run, context):
    async def work():
        async with context.begin_process("test") as process:
            process: IChatBioAgentProcess
            await process.create_artifact(
                mimetype="text/plain",
                description="test artifact",
                uris=["https://test.artifact"],
                metadata={"source": "nowhere"},
            )

    messages = await run(work())
    assert messages == [
        ProcessBeginResponse(summary="test"),
        ArtifactResponse(
            mimetype="text/plain",
            description="test artifact",
            uris=["https://test.artifact"],
            metadata={"source": "nowhere"},
        ),
    ]
