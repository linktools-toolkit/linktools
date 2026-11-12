#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recovery templates retain shared Step leaves through publication and relocation."""

import pytest
from pathlib import Path
from dataclasses import replace
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from linktools.ai.core import ExecutionStatus,ExecutionEventType,StopReason,UsageMetrics,ToolOperationStatus
from linktools.ai.errors import ErrorCode
from linktools.ai.execution import ExecutionRecoveryDetail,PendingToolContinuation,PendingDeferredCall,RecoveryTerminalOutcome,RecoveryTerminalHandoff,RecoveryHandoffPhase
from linktools.ai.persistence import Persistence,PersistenceStorage
from linktools.ai.persistence.contracts import ToolOperationRecord
from linktools.ai.persistence.codec import decode_envelope,payload_reference,tool_payload_reference
from linktools.ai.storage import StoredPayload,InMemoryObjectStore
from linktools.ai.persistence._snapshot import SnapshotLimits
from .test_step_projection import _execution
class Guard:
    @asynccontextmanager
    async def offline_exclusivity(self) -> AsyncIterator[None]:
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("pending", "handoff"))
async def test_recovery_templates_share_leaf_owners_and_restore(tmp_path: Path, kind: str) -> None:
    storage=PersistenceStorage.sqlite(tmp_path/'db');objects=InMemoryObjectStore();limits=SnapshotLimits(10000,20_000_000)
    current=_execution();args=StoredPayload.inline_json({'q':'shared tool arguments'})
    if kind=='pending':
        pending=PendingToolContinuation('run',approvals=(PendingDeferredCall('call','tool',args),))
        current=replace(current,status=ExecutionStatus.WAITING_DEFERRED,recovery=ExecutionRecoveryDetail(agent_run_id='run',pending_tools=pending))
    else:
        details={'reason':'shared diagnostics'}
        outcome=RecoveryTerminalOutcome(ExecutionStatus.FAILED,ErrorCode.EXECUTION_FAILED.value,details,StopReason.ERROR,None,None,UsageMetrics(),ExecutionEventType.EXECUTION_FAILED,{},current.updated_at)
        current=replace(current,status=ExecutionStatus.FAILED,error_code=ErrorCode.EXECUTION_FAILED.value,safe_error_details=details,recovery=ExecutionRecoveryDetail(agent_run_id='run',handoff_phase=RecoveryHandoffPhase.PREPARED,terminal_handoff=RecoveryTerminalHandoff(outcome,'run',None)))
    async with Persistence.open('native-result',tenant_id='tenant',storage=storage) as state:
        repo=state.execution.executions;await repo.create_execution(current)
        raw=await repo.state_store.read_record(repo.execution_record_key('execution'));fields=decode_envelope(raw.data).value['payload']['fields']
        if kind=='pending':
            tool=ToolOperationRecord('operation','execution','run','call','tool',args.digest,'binding',True,ToolOperationStatus.PENDING,None,0,None,current.created_at,current.updated_at,arguments_payload=args)
            await state.checkpoint.tools.reserve(tool)
            toolraw=await state.checkpoint.tools.state_store.read_record(state.checkpoint.tools._tool_key('operation'))
            toolfields=decode_envelope(toolraw.data).value['payload']['fields']
            assert payload_reference(fields['recovery']['fields']['pending_tools']['fields']['arguments']['0'])==tool_payload_reference(toolfields['arguments_payload'])
            assert await state.checkpoint.tools.get_operation('operation',tenant_id='tenant')==tool
        else:
            assert payload_reference(fields['terminal_details'])==payload_reference(fields['recovery']['fields']['terminal_handoff']['fields']['terminal_details'])
        original_scope=payload_reference(fields['terminal_details'])[0]
        assert await repo.get_recovery('execution',tenant_id='tenant')==current
        assert (await repo.list_recoverable_page(tenant_id='tenant',cursor=None,limit=10)).items==(current,)
        await state.purge_unreferenced_objects((),exclusive=Guard())
    async with Persistence.open('native-result',tenant_id='tenant',storage=storage,read_only=True) as reader:snapshot=await reader.export_snapshot(object_store=objects,limits=limits)
    target=tmp_path/'restored';await Persistence.restore_snapshot(snapshot,object_store=objects,root=target,limits=limits)
    async with Persistence.open('native-result',tenant_id='tenant',storage=PersistenceStorage.from_root(target)) as state:
        repo=state.execution.executions;assert await repo.get_recovery('execution',tenant_id='tenant')==current
        restored_raw=await repo.state_store.read_record(repo.execution_record_key('execution'))
        restored_fields=decode_envelope(restored_raw.data).value['payload']['fields']
        assert payload_reference(restored_fields['terminal_details'])[0] != original_scope
        if kind=='pending':assert (await state.checkpoint.tools.get_operation('operation',tenant_id='tenant')).arguments_payload==args
        else:
            completed=replace(current,revision=1,recovery=replace(current.recovery,handoff_phase=RecoveryHandoffPhase.COMPLETED))
            await repo.compare_and_swap('execution',tenant_id='tenant',expected_revision=0,next_record=completed)
            assert (await repo.list_recoverable_page(tenant_id='tenant',cursor=None,limit=10)).items==()
        await state.purge_unreferenced_objects((),exclusive=Guard())
