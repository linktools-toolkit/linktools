#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepared requests reuse equivalent durable tool-result frontiers."""

import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder
from linktools.ai.runtime._run_hooks import AgentRunCapture
from linktools.ai.persistence.runs import AgentRunCheckpoint, StagingAgentRunStore


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    ("none", "instructions", "parts", "prefix", "projection", "timestamp", "metadata", "request_state"),
)
async def test_native_frontier_preserves_prepared_request_without_mutating_prior_snapshot(change: str) -> None:
    from copy import deepcopy
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage
    from tests.ai._runtime_test_helpers import native_checkpoint

    checkpoint = native_checkpoint()
    first = ModelRequest(parts=[UserPromptPart("work")])
    response = ModelResponse(parts=[ToolCallPart("work", {}, "call")])
    pending = ModelRequest(parts=[ToolReturnPart("work", "original", "call")])
    ctx = RunContext(
        deps=None, model=TestModel(), usage=RunUsage(),
        messages=[first, response, pending], run_id="run",
        conversation_id="conversation", run_step=1,
        _event_stream_buffer=[], _capability=checkpoint,
    )
    await checkpoint.before_run(ctx)
    await checkpoint.before_model_request(ctx, SimpleNamespace())
    previous = await checkpoint.store.latest_snapshot(run_id="run", include_interrupted=True)
    original = deepcopy(ctx.messages)
    assert previous is not None and previous.messages == original
    ctx.run_step = 2
    if change == "instructions":
        pending.instructions = "new repository instructions"
    elif change == "parts":
        pending.parts.append(UserPromptPart("new queued content"))
    elif change == "prefix":
        first.parts.append(UserPromptPart("changed history"))
    elif change == "projection":
        ctx.messages[0] = ModelRequest(parts=[UserPromptPart("summary")])
    elif change == "timestamp":
        pending.timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    elif change == "metadata":
        pending.metadata = {"new": "fact"}
    elif change == "request_state":
        pending.state = "interrupted"
    await checkpoint.before_model_request(ctx, SimpleNamespace())
    latest = await checkpoint.store.latest_snapshot(run_id="run", include_interrupted=True)
    assert latest is not None and latest.messages == ctx.messages
    assert previous.messages == original
    assert (latest.messages == original) is (change == "none")


_SCRIPT = r"""
import asyncio, os, sys
from pathlib import Path
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, ToolReturn, ImageUrl, UserPromptPart
from linktools.ai.capability import AgentContext, ExecutionCapability, ToolCallFailed
from pydantic_ai_harness.step_persistence import StepPersistence
from tests.ai._runtime_test_helpers import native_checkpoint
from linktools.ai.core import ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from tests.ai._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
root_path, mode, policy, capture_frontier = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == '1'
payloads = ('scalar', 'media', 'failure')
ready = set()
async def run_payload(payload):
    root = root_path / payload
    if mode == 'create':
        root.mkdir()
    async def work(ctx: AgentContext[None]) -> ToolReturn:
        with (root / 'calls').open('a') as output:
            output.write(mode + '\n')
            output.flush()
            os.fsync(output.fileno())
        if payload == 'failure':
            raise ToolCallFailed('original business failure')
        return ToolReturn(return_value={'receipt': 'original'}, content=[ImageUrl(url='https://example.com/original-evidence.png')] if payload == 'media' else None, metadata={'origin':'original'})
    async def model(messages, info):
        returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
        with (root / 'models').open('a') as output:
            output.write(mode + ':' + ('result' if returns else 'request') + '\n')
        if not returns:
            assert mode == 'create', 'must recover original tool call without another provider request'
            return ModelResponse([ToolCallPart('work', {}, 'original-call')])
        if mode == 'create':
            assert payload not in ready, ('duplicate frontier', payload)
            ready.add(payload)
            print('FRONTIER_READY', payload, flush=True)
            if ready == set(payloads):
                os._exit(73)
            await asyncio.Event().wait()
        part = returns[-1]
        assert part.tool_call_id == 'original-call'
        if payload == 'failure':
            assert part.outcome == 'failed' and 'original business failure' in str(part.content), part
        else:
            assert part.content == {'receipt': 'original'}, part
            assert part.metadata == {'origin': 'original'}, part
        if payload == 'media':
            images = [c for m in messages for p in m.parts if isinstance(p, UserPromptPart) and not isinstance(p.content,str) for c in p.content if isinstance(c,ImageUrl)]
            assert len(images) == 1 and images[0].url == 'https://example.com/original-evidence.png'
        return ModelResponse([TextPart('complete')])
    async def main():
        features = RuntimeFeatures('coalesce-crash')
        checkpoint = native_checkpoint()
        checkpoint.capture_frontier = capture_frontier
        features.capability(checkpoint,scope='runtime')
        features.capability(ExecutionCapability(retain_history=True),scope='runtime')
        features.tool(work,effect_policy=policy)
        features.agent('default',allow_tools=('work',))
        binding=_Models(_UsageFunctionModel(model))
        binding.contract={**binding.contract,'input_mime_types': ['image/png']}
        async with Persistence.open('coalesce-crash', storage=PersistenceStorage.sqlite(root/'state.db')) as persistence, Runtime.attach(persistence=persistence,features=model_features(binding,features),auto_recover=False) as runtime:
            if mode == 'create':
                session = await runtime.agents.get().create_session('session')
                execution = await session.start('inspect',idempotency_key='one')
                (root/'execution').write_text(execution.execution_id)
                await execution.wait(timeout_seconds=10)
                raise AssertionError('exit not reached')
            execution_id=(root/'execution').read_text()
            operations=await persistence.checkpoint.tools.list_by_execution(execution_id,tenant_id=runtime.tenant_id)
            assert len(operations) == (0 if policy=='none' else 1)
            await runtime.executions.recover(execution_id,principal=runtime.default_principal,idempotency_key='recover')
            execution=await runtime.executions.get(execution_id)
            result=await execution.wait(timeout_seconds=10)
            assert result.result.status is ExecutionStatus.SUCCEEDED,result
            assert result.result.output == {'text':'complete'}
            assert (root/'calls').read_text().splitlines() == ['create']
            assert (root/'models').read_text().splitlines() == ['create:request','create:result','resume:result']
            print('EXACT_RESULT_RESTORED',policy,payload,capture_frontier,flush=True)
    try:
        await main()
    except Exception as error:
        raise AssertionError((mode, policy, payload, capture_frontier)) from error
async def main():
    if mode == 'create':
        # Each database remains open at its prepared frontier until all three
        # payloads are ready for the same abrupt process death.
        await asyncio.gather(*(run_payload(payload) for payload in payloads))
    else:
        for payload in payloads:
            await run_payload(payload)
asyncio.run(main())
"""


@pytest.mark.parametrize("policy", ["none", "replay_safe", "non_replay_safe"])
@pytest.mark.parametrize("capture_frontier", [False, True])
@pytest.mark.merge
def test_frontier_process_exit_recovers_original_result(
    tmp_path: Path,
    policy: str,
    capture_frontier: bool,
) -> None:
    args = [sys.executable, "-c", _SCRIPT, str(tmp_path)]
    created = subprocess.run(
        [*args, "create", policy, str(int(capture_frontier))],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert created.returncode == 73, (created.stdout, created.stderr)
    assert sorted(line for line in created.stdout.splitlines() if line.startswith("FRONTIER_READY")) == [
        f"FRONTIER_READY {payload}" for payload in ("failure", "media", "scalar")
    ]
    resumed = subprocess.run(
        [*args, "resume", policy, str(int(capture_frontier))],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert resumed.returncode == 0, (resumed.stdout, resumed.stderr)
    assert [line for line in resumed.stdout.splitlines() if line.startswith("EXACT_RESULT_RESTORED")] == [
        f"EXACT_RESULT_RESTORED {policy} {payload} {capture_frontier}"
        for payload in ("scalar", "media", "failure")
    ]
