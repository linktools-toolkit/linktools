#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session occupancy consumes membership, not executable content."""

from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionStatus, Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.persistence._execution_content import ExecutionContent

from .test_session_timeline import _execution, _session


@pytest.mark.asyncio
@pytest.mark.parametrize('backend', ('memory', 'sqlite'))
@pytest.mark.parametrize('case', ('active', 'terminal', 'missing', 'other-session', 'child', 'foreign-tenant'))
async def test_inspection_keeps_admission_validation_without_payload_reads(tmp_path, monkeypatch, backend, case):
    storage = PersistenceStorage.in_memory() if backend == 'memory' else PersistenceStorage.sqlite(tmp_path / 'state.db')
    async with Persistence.open('inspect', tenant_id='tenant', storage=storage) as state:
        execution = _execution('active', status=ExecutionStatus.FAILED if case == 'terminal' else ExecutionStatus.STARTED, prompt='private body')
        if case == 'other-session':
            execution = replace(execution, session_id='other')
        elif case == 'child':
            # Simulate malformed membership independently of authoring lineage constraints.
            object.__setattr__(execution, 'parent_execution_id', 'parent')
        if case != 'missing':
            await state.execution.executions.create(execution)
        await state.conversation.sessions.create(replace(_session(), active_execution_id='active'))

        async def no_details(*args, **kwargs):
            pytest.fail('occupancy must not restore executable content')

        monkeypatch.setattr(ExecutionRepositoryImpl, 'get', no_details)
        monkeypatch.setattr(ExecutionContent, 'expand_policies', no_details)
        monkeypatch.setattr(ExecutionContent, 'read_payloads', no_details)
        async with History.open(persistence=state) as history:
            principal = Principal('owner', 'other' if case == 'foreign-tenant' else 'tenant')
            if case in {'missing', 'other-session', 'child', 'foreign-tenant'}:
                with pytest.raises(AIError) as caught:
                    await history.inspect_session('session', principal=principal)
                assert caught.value.code is (ErrorCode.AUTHORIZATION_DENIED if case == 'foreign-tenant' else ErrorCode.STORAGE_INTEGRITY_ERROR)
            else:
                view = await history.inspect_session('session', principal=principal)
                assert view.active_execution_id == (None if case == 'terminal' else 'active')
