#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed failure events remain readable through offline snapshots and cleanup."""
import pytest
from linktools.ai.storage import InMemoryObjectStore
from linktools.ai.persistence.snapshots import SnapshotLimits
from tests.ai.test_native_result_portable_gc import Guard
from pathlib import Path
from sqlalchemy import text
from linktools.ai.errors import AIError
from pydantic_ai_harness.step_persistence import StepEvent
from linktools.ai.persistence import Persistence,PersistenceStorage,SqlStepStore,NativeToolErrorReference
from linktools.ai.persistence.codec import decode_envelope,native_tool_error_reference
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.core import canonical_json_bytes,deterministic_id
from linktools.ai.errors import ErrorCode
from linktools.ai.storage import StoredPayload
from tests.ai.test_tool_execution_boundary import _admit,_commands
@pytest.mark.asyncio
async def test_fixed_error_survives_readonly_snapshot_and_gc(tmp_path: Path) -> None:
 config=PersistenceStorage.sqlite(tmp_path/'e.db')
 for split in (False,):
  namespace='tool-boundary' if not split else 'tool-boundary-split'
  async with Persistence.open(namespace,tenant_id='tenant',storage=config) as state:
   receipt=await _admit(state)
   tools=state.checkpoint.tools
   native=SqlStepStore.for_state_store(tools.state_store,namespace=namespace,tenant_id='tenant')
   scope,store=await tools.state_store.storage_group.open_archive(create=True)
   native.bind_message_archive(ContentArchive(store,scope,namespace=namespace,tenant_id='tenant',object_store=None))
   event=StepEvent(run_id='run',kind='tool_call_failed',step_index=1,tool_name='tool',tool_call_id='call',error='failure')
   from dataclasses import replace
   event=replace(event,idempotency_key=deterministic_id('tool','operation',1,receipt.operation.value.fence,'tool_call_failed'))
   ref=NativeToolErrorReference(scope,native.store_key,event.run_id,event.idempotency_key)
   payload=StoredPayload.inline_bytes(canonical_json_bytes({'version':1,'kind':'tool_call_failed','message':'failure'}))
   if split:
    await tools.fail_payload('operation',tenant_id='tenant',owner='worker',fence=1,error_code=ErrorCode.TOOL_EXECUTION_FAILED.value,error_payload=payload)
    await native.append_event(event, fixed_error_source=True)
    await tools.share_published_error('operation',tenant_id='tenant',source=ref)
   else:
    prepared=await native.prepare_step(events=(event,), fixed_error_sources=True)
    async def write(group: object) -> None:
     await native.write_prepared_step(group.transaction(tools.state_store),prepared)
    settled=await _commands(state).commit_tool_terminal_boundary('operation',tenant_id='tenant',owner='worker',fence=1,execution=receipt.execution,observed=receipt.operation,error_code=ErrorCode.TOOL_EXECUTION_FAILED.value,error_payload=payload,error_source=ref,write=write,stores=(tools.state_store,))
   records=await tools.state_store.read(lambda tx:tx.scan_records())
   raw=next(r for r in records if r.kind=='tool_operation')
   assert native_tool_error_reference(decode_envelope(raw.data).value['payload']['fields']['error_payload'])==ref
   assert (await tools.get_operation('operation',tenant_id='tenant')).error_payload==payload

  async with Persistence.open(namespace,tenant_id='tenant',storage=config,read_only=True) as state:
   assert (await state.checkpoint.tools.get_operation('operation',tenant_id='tenant')).error_payload==payload
   assert (await state.checkpoint.tools.get_operations(('operation',),tenant_id='tenant'))[0].error_payload==payload
   objects=InMemoryObjectStore('fixed-error-snapshot')
   limits=SnapshotLimits(10000,20_000_000)
   snapshot=await state.export_snapshot(object_store=objects,limits=limits)
  target=tmp_path/'restored'
  await Persistence.restore_snapshot(snapshot,object_store=objects,root=target,limits=limits)
  restored=PersistenceStorage.from_root(target)
  async with Persistence.open(namespace,tenant_id='tenant',storage=restored) as state:
   assert (await state.checkpoint.tools.get_operation('operation',tenant_id='tenant')).error_payload==payload
   await state.purge_unreferenced_objects((),exclusive=Guard())
   assert (await state.checkpoint.tools.get_operation('operation',tenant_id='tenant')).error_payload==payload

  async with Persistence.open(namespace,tenant_id='tenant',storage=restored) as state:
   context=state.checkpoint.tools.state_store.storage_group.context
   async with context.engine.begin() as connection:
    await connection.execute(text("DELETE FROM ai_step_events WHERE kind='tool_call_failed'"))
  async with Persistence.open(namespace,tenant_id='tenant',storage=restored,read_only=True) as state:
   with pytest.raises(AIError) as missing:
    await state.checkpoint.tools.get_operation('operation',tenant_id='tenant')
   assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
