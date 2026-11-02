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
        import asyncio, os, sys
        from pathlib import Path
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
        from linktools.ai.capability import AgentContext, ExecutionCapability
        from linktools.ai.core import CommitPolicy, ExecutionStatus
        from linktools.ai.persistence import Persistence, PersistenceStorage
        from linktools.ai.storage import InMemoryCacheBackend
        from linktools.ai.runtime import Runtime, RuntimeFeatures
        from tests.ai._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
        root, mode = Path(sys.argv[1]), sys.argv[2]
        entered = asyncio.Event()
        async def save(ctx: AgentContext[None], value: int) -> str:
            (root / 'effect').write_text(str(value))
            with (root / 'calls').open('a') as stream:
                stream.write(mode + '\\n')
            if mode == 'create':
                entered.set()
                await asyncio.Event().wait()
            return 'saved ' + str(value)
        async def model(messages, info):
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
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
                if mode == 'retry':
                    assert await state.execution.executions.get((root / 'old-id').read_text(), tenant_id=runtime.tenant_id) is None
                session = (await runtime.agents.get().create_session('session') if mode == 'create'
                    else await runtime.sessions.get('session', principal=runtime.default_principal))
                handle = await session.start('save seven', idempotency_key=mode)
                if mode == 'create':
                    await asyncio.wait_for(entered.wait(), 5)
                    assert await state.execution.executions.get(handle.execution_id, tenant_id=runtime.tenant_id) is None
                    assert not await state.checkpoint.tools.list_by_execution(handle.execution_id, tenant_id=runtime.tenant_id)
                    (root / 'old-id').write_text(handle.execution_id)
                    os._exit(73)
                assert handle.execution_id != (root / 'old-id').read_text()
                assert (await handle.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
                assert not await state.checkpoint.tools.list_by_execution(handle.execution_id, tenant_id=runtime.tenant_id)
                assert (root / 'effect').read_text() == '7'
                assert (root / 'calls').read_text().splitlines() == ['create', 'retry']
        asyncio.run(main())
    ''')
    for mode, expected in (("create", 73), ("retry", 0)):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path), mode],
                                env=dict(os.environ), capture_output=True, text=True, timeout=25)
        assert result.returncode == expected, result.stdout + result.stderr
