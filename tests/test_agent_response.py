import asyncio
from typing import Coroutine

import pytest
from pydantic import BaseModel

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
    DirectResponseAck,
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
    async def do_work(work: Coroutine, followup=None) -> list[ResponseMessage]:
        done = object()

        async def work_and_finish():
            await work
            await channel.submit(done)

        task = asyncio.create_task(work_and_finish())

        messages = []
        while True:
            async with channel.receive() as message:
                if message is done:
                    break
                messages.append(message)

            match message:
                case ArtifactResponse():
                    await channel.submit(ArtifactAck(None))
                case DirectResponse(response_model=response_model):
                    if response_model is not None:
                        await channel.submit(
                            DirectResponseAck(
                                explanation="test value", value=followup, complete=True
                            )
                        )

        await task

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
async def test_submit_direct_response_with_question(run, context):
    class TrueOrFalse(BaseModel):
        answer: bool

    answer_box = [None]

    async def work():
        response = await context.reply("hi", response_model=TrueOrFalse)
        answer_box[0] = response.value

    messages = await run(work(), followup=TrueOrFalse(answer=True))

    assert messages == [DirectResponse(text="hi", response_model=TrueOrFalse)]
    assert answer_box[0] == TrueOrFalse(answer=True)


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
