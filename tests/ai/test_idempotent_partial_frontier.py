#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A checkpoint may lag a genuinely published result from a parallel tool group."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest


@pytest.mark.parametrize("rich", (False, True))
def test_recovery_preserves_published_partial_tool_result(tmp_path: Path, rich: bool) -> None:
    script = textwrap.dedent('''
        import asyncio, os, sys
        from pathlib import Path
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, ToolReturn
        from linktools.ai.capability import AgentContext, CheckpointCapability, ExecutionCapability, TraceCapability
        from linktools.ai.core import CommitPolicy, ExecutionStatus
        from linktools.ai.persistence import Persistence, PersistenceStorage, PersistenceDomain
        from linktools.ai.runtime import Runtime, RuntimeFeatures
        from linktools.ai.runtime import _tool as tool_owner
        tool_owner.TOOL_OPERATION_LEASE_SECONDS = 1
        from tests.ai._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
        root, mode = Path(sys.argv[1]), sys.argv[2]
        rich = sys.argv[3] == "rich"
        pending = asyncio.Event()
        async def first(ctx: AgentContext[None]) -> str:
            path = root / "first-invocations"
            count = int(path.read_text()) + 1 if path.exists() else 1
            path.write_text(str(count))
            (root / "business-effect").write_text("7")
            value = "observed value " + str(count)
            return ToolReturn(return_value=value, content="extra native evidence", metadata={"source": "first"}) if rich else value
        async def second(ctx: AgentContext[None]) -> str:
            if mode == "create":
                pending.set()
                await asyncio.Event().wait()
            return "second result"
        async def model(messages, info):
            returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
            if returned:
                assert any(p.tool_call_id == "a" and p.content == "observed value 1" for p in returned)
                if rich:
                    assert "extra native evidence" in str(messages)
                    assert next(p for p in returned if p.tool_call_id == "a").metadata == {"source": "first"}
                print("FINAL_MODEL_RESULTS", repr(returned), flush=True)
                return ModelResponse([TextPart("done")])
            assert mode == "create", "resume must not regenerate calls"
            return ModelResponse([ToolCallPart("first", {}, "a"), ToolCallPart("second", {}, "b")])
        async def main():
            group = RuntimeFeatures("partial-retry")
            group.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()))
            group.capability(CheckpointCapability())
            group.capability(TraceCapability())
            group.tool(first)
            group.tool(second)
            group.agent("default", allow_tools=("first", "second"))
            async with Persistence.open("partial-retry", storage=PersistenceStorage.sqlite(root / "source.db")) as state, Runtime.attach(
                persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), group),
                auto_recover=False, metrics=None,
            ) as runtime:
                if mode == "create":
                    handle = await (await runtime.agents.get().create_session("session")).start("do both")
                    await asyncio.wait_for(pending.wait(), 5)
                    trace = runtime._run_store.read_store(PersistenceDomain.TRACE)
                    checkpoints = runtime._run_store.read_store(PersistenceDomain.CHECKPOINT)
                    async def published():
                        while True:
                            runs = await trace.list_agent_runs()
                            if runs:
                                run = runs[0]
                                head = await trace.transcript_repository.get_head(run.agent_run_id)
                                message, _ = await trace.transcript_repository.load_pending(head) if head else (None, ())
                                if message and any(isinstance(p, ToolReturnPart) and p.tool_call_id == "a" and p.content == "observed value 1" for p in message.parts):
                                    checkpoint = await checkpoints.latest_checkpoint(agent_run_id=run.agent_run_id, include_interrupted=True)
                                    assert checkpoint is not None
                                    assert not any(isinstance(p, ToolReturnPart) and p.tool_call_id == "a" for m in checkpoint.messages for p in m.parts)
                                    return
                            await asyncio.sleep(0.01)
                    await asyncio.wait_for(published(), 5)
                    operations = await state.checkpoint.tools.list_by_execution(handle.execution_id, tenant_id=runtime.tenant_id)
                    assert len(operations) == 2 and all(operation.replay_safe for operation in operations)
                    (root / "execution").write_text(handle.execution_id)
                    print("TRACE_A_PUBLISHED_CHECKPOINT_BEFORE_TOOL_GROUP", flush=True)
                    os._exit(73)
                execution_id = (root / "execution").read_text()
                await runtime.executions.recover(execution_id, principal=runtime.default_principal, idempotency_key="replacement")
                result = (await (await runtime.executions.get(execution_id)).wait(timeout_seconds=10)).result
                print("RECOVERY_RESULT", result, "A_INVOCATIONS", (root / "first-invocations").read_text(), flush=True)
                assert result.status is ExecutionStatus.SUCCEEDED, result
                assert (root / "business-effect").read_text() == "7"
                assert (root / "first-invocations").read_text() == "1"
        asyncio.run(main())
    ''')
    for mode, code in (("create", 73), ("resume", 0)):
        if mode == "resume":
            time.sleep(1.1)
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path), mode, "rich" if rich else "plain"],
                                env=dict(os.environ), capture_output=True, text=True, timeout=25)
        print(mode, result.stdout, result.stderr, flush=True)
        assert result.returncode == code
