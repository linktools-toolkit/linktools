#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal-only failure permits a new turn, not exact execution recovery."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_terminal_worker_exit_retries_idempotent_effect_as_new_turn(tmp_path: Path) -> None:
    script = textwrap.dedent('''
        import asyncio, json, os, sys
        from pathlib import Path
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
        from linktools.ai.capability import AgentContext, ExecutionCapability
        from linktools.ai.core import CommitPolicy, ExecutionStatus, ToolOperationStatus
        from linktools.ai.model.messages import decode_model_messages
        from linktools.ai.persistence import Persistence, PersistenceStorage
        from linktools.ai.storage import InMemoryCacheBackend
        from linktools.ai.runtime import Runtime, RuntimeFeatures
        from tests.ai._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
        root, mode = Path(sys.argv[1]), sys.argv[2]
        entered = asyncio.Event()
        async def save(ctx: AgentContext[None], value: int) -> str:
            assert mode != 'replay', 'a confirmed retry must not perform another effect'
            (root / 'effect').write_text(str(value))
            with (root / 'calls').open('a') as stream:
                stream.write(mode + '\\n')
            if mode == 'create':
                entered.set()
                await asyncio.Event().wait()
            return 'saved ' + str(value)
        async def model(messages, info):
            assert mode != 'replay', 'a confirmed retry must not call the model again'
            returned = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
            with (root / 'model-calls').open('a') as stream:
                stream.write(mode + (':result' if returned else ':request') + '\\n')
            if returned:
                assert len(returned) == 1 and returned[0].content == 'saved 7'
                return ModelResponse([TextPart('done')])
            return ModelResponse([ToolCallPart('save', {'value': 7}, 'save-call')])
        async def main():
            features = RuntimeFeatures('terminal-retry')
            features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()))
            features.tool(save)
            features.agent('default', allow_tools=('save',))
            async with Persistence.open('terminal-retry', storage=PersistenceStorage.sqlite(
                root / 'state.db', cache=InMemoryCacheBackend(max_bytes=4*1024*1024),
                coordination=InMemoryCacheBackend(max_bytes=4*1024*1024),
            )) as state, Runtime.attach(persistence=state,
                features=model_features(_Models(_UsageFunctionModel(model)), features),
                metrics=None, auto_recover=False,
            ) as runtime:
                if mode != 'create':
                    assert await state.execution.executions.get((root / 'old-id').read_text(), tenant_id=runtime.tenant_id) is None
                session = (await runtime.agents.get().create_session('session') if mode == 'create'
                    else await runtime.sessions.get('session', principal=runtime.default_principal))
                handle = await session.start('save seven', idempotency_key='create' if mode == 'create' else 'retry')
                if mode == 'create':
                    await asyncio.wait_for(entered.wait(), 5)
                    assert await state.execution.executions.get(handle.execution_id, tenant_id=runtime.tenant_id) is None
                    assert not await state.checkpoint.tools.list_by_execution(handle.execution_id, tenant_id=runtime.tenant_id)
                    (root / 'old-id').write_text(handle.execution_id)
                    os._exit(73)
                assert handle.execution_id != (root / 'old-id').read_text()
                result = (await handle.wait(timeout_seconds=10)).result
                assert result.status is ExecutionStatus.SUCCEEDED and result.output == {'text': 'done'}
                repository = state.execution.executions
                execution = await repository.state_store.read(lambda _tx: repository.get(
                    handle.execution_id, tenant_id=runtime.tenant_id))
                assert execution is not None and execution.status is ExecutionStatus.SUCCEEDED
                assert not execution.restart_checkpoints
                operation, = await state.checkpoint.tools.list_by_execution(handle.execution_id, tenant_id=runtime.tenant_id)
                assert operation.status is ToolOperationStatus.COMPLETED and operation.replay_safe
                assert operation.tool_name == 'save' and operation.tool_call_id == 'save-call'
                assert operation.owner is not None and operation.owner == execution.producer_claim_id
                assert operation.fence > 0 and operation.producer_generation == execution.producer_generation
                assert operation.result_payload is not None
                payload = operation.result_payload.decode()
                assert isinstance(payload, bytes)
                messages = decode_model_messages(payload)
                assert len(messages) == 1 and isinstance(messages[0], ModelRequest)
                assert len(messages[0].parts) == 1
                returned = messages[0].parts[0]
                assert isinstance(returned, ToolReturnPart)
                assert returned.tool_call_id == operation.tool_operation_id and returned.content == 'saved 7'
                receipt = {
                    'execution_id': handle.execution_id, 'operation_id': operation.tool_operation_id,
                    'owner': operation.owner, 'fence': operation.fence,
                    'producer_generation': operation.producer_generation, 'payload': payload.hex(),
                }
                if mode == 'retry':
                    (root / 'retry-receipt').write_text(json.dumps(receipt))
                else:
                    assert receipt == json.loads((root / 'retry-receipt').read_text())
                assert (root / 'effect').read_text() == '7'
                assert (root / 'calls').read_text().splitlines() == ['create', 'retry']
                assert (root / 'model-calls').read_text().splitlines() == ['create:request', 'retry:request', 'retry:result']
        asyncio.run(main())
    ''')
    for mode, expected in (("create", 73), ("retry", 0), ("replay", 0)):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path), mode],
                                env=dict(os.environ), capture_output=True, text=True, timeout=25)
        assert result.returncode == expected, result.stdout + result.stderr
