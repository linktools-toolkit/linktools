#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution control updates retain immutable input and output bodies by reference."""

from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import archive_scope_token, ContentArchive, ContentRecord
from linktools.ai.persistence._archive_closure import iter_archive_content_refs, rekey_archive_roots
from linktools.ai.persistence._execution_content import expand_snapshot_execution_content, snapshot_execution_content_size
from linktools.ai.persistence.codec import _decode_enveloped_domain
from linktools.ai.persistence.contracts import ExecutionRecord
from linktools.ai.persistence.state import RecordQuery
from linktools.ai.storage import StoredPayload

from .test_incremental_projection_boundaries import _storage
from .test_session_timeline import _execution


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "local_sqlite", "sqlite"))
async def test_large_payloads_survive_repeated_control_cas_without_body_reads_or_reencoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="payloads", tenant_id="tenant")
    try:
        repository = state.execution.executions
        value = _execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="large input " * 20000)
        value = replace(value, result=replace(value.result, output=StoredPayload.inline_json({"body": "large output " * 20000})),
            binding=replace(value.binding, output_schema={"type": "string", "description": "large schema " * 20000}))
        await repository.create_execution(value)
        raw = await repository.state_store.read_record(repository.execution_record_key(value.execution_id))
        assert len(canonical_json_bytes(raw.data)) < 12000
        refs = tuple(iter_archive_content_refs(raw.data))
        assert len(refs) == 3
        _, archive = repository.state_store.storage_group.opened_archive
        before = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        assert len(before) == 3
        original = ContentArchive.read_contents

        async def reject_body(*args, **kwargs):
            raise AssertionError("control-only CAS must not read immutable body")

        monkeypatch.setattr(ContentArchive, "read_contents", reject_body)
        import linktools.ai.persistence._execution_content as content_module
        encode = content_module.canonical_json_bytes

        def no_body_encoding(item):
            if len(encode(item)) > 65536:
                raise AssertionError("unchanged immutable payload and binding must reuse their encoding")
            return encode(item)

        monkeypatch.setattr(content_module, "canonical_json_bytes", no_body_encoding)
        for _ in range(2):
            changed = replace(value, revision=value.revision + 1, event_seq=value.event_seq + 1)
            await repository.compare_and_swap(value.execution_id, tenant_id="tenant",
                expected_revision=value.revision, next_record=changed)
            value = changed
            row = await repository.state_store.read_record(repository.execution_record_key(value.execution_id))
            assert tuple(iter_archive_content_refs(row.data)) == refs
            assert len(canonical_json_bytes(row.data)) < 12000
        after = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        assert after == before
        monkeypatch.setattr(ContentArchive, "read_contents", original)
        monkeypatch.setattr(content_module, "canonical_json_bytes", encode)
        assert await repository.get_fresh(value.execution_id, tenant_id="tenant") == value
        assert (await repository.get_timeline_many((value.execution_id,), tenant_id="tenant"))[value.execution_id].result == value.result
        with pytest.raises(AIError) as immutable:
            await repository.compare_and_swap(value.execution_id, tenant_id="tenant", expected_revision=value.revision,
                next_record=replace(value, revision=value.revision + 1,
                    result=replace(value.result, output=StoredPayload.inline_json("changed " * 20000))))
        assert immutable.value.code is ErrorCode.EXECUTION_RESULT_CONFLICT
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_payload_reference_snapshot_scope_integrity_and_budget(tmp_path: Path) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="payloads", tenant_id="tenant")
    try:
        repository = state.execution.executions
        value = _execution("execution", status=ExecutionStatus.STARTED, prompt="body " * 20000)
        await repository.create_execution(value)
        record = await repository.state_store.read_record(repository.execution_record_key(value.execution_id))
        scope, archive = repository.state_store.storage_group.opened_archive
        rows = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        contents = {(scope, archive_scope_token(row.scope_key), row.sort_key.decode("utf-8")): ContentRecord.from_record(row) for row in rows}
        assert snapshot_execution_content_size(record, contents) == sum(item.raw_size for item in contents.values())
        semantic = expand_snapshot_execution_content(record, contents, namespace="payloads", tenant_id="tenant")
        assert _decode_enveloped_domain(semantic.data, ExecutionRecord) == value
        remapped = replace(record, data=rekey_archive_roots(dict(record.data), {scope: "target"}))
        relocated = {("target", *key[1:]): body for key, body in contents.items()}
        assert expand_snapshot_execution_content(remapped, relocated, namespace="payloads", tenant_id="tenant") == semantic
        for malformed in ({}, {key: replace(body, body=b"bad") for key, body in contents.items()}):
            with pytest.raises(AIError) as invalid:
                expand_snapshot_execution_content(record, malformed, namespace="payloads", tenant_id="tenant")
            assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError):
            expand_snapshot_execution_content(record, contents, namespace="payloads", tenant_id="other")
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("corruption", ("missing", "body"))
async def test_payload_publication_rolls_back_and_complete_reads_require_present_valid_bodies(tmp_path: Path, backend: str, corruption: str) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="payloads", tenant_id="tenant")
    try:
        repository = state.execution.executions
        value = _execution("execution", status=ExecutionStatus.STARTED, prompt="body " * 20000)
        await repository.initialize_content(create=True)

        async def abort(transaction):
            await repository.create_execution_in_transaction(transaction, value)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError) as aborted:
            await repository.state_store.mutate(abort)
        assert aborted.value.code is ErrorCode.STORAGE_CONFLICT
        assert await repository.get_fresh(value.execution_id, tenant_id="tenant") is None
        _, archive = repository.state_store.storage_group.opened_archive
        assert not await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        await repository.create_execution(value)
        assert await repository.get_fresh(value.execution_id, tenant_id="tenant") == value
        rows = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        body = next(row for row in rows if row.data["codec"] == "execution-payload")
        if corruption == "missing":
            await archive.mutate(lambda tx: tx.delete_record(body.key, expected_storage_revision=body.storage_revision))
        else:
            await archive.mutate(lambda tx: tx.replace_record(replace(body, payload_bytes=b"corrupt", storage_revision=body.storage_revision + 1),
                expected_storage_revision=body.storage_revision))
        with pytest.raises(AIError) as invalid:
            await repository.get_fresh(value.execution_id, tenant_id="tenant")
        assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_large_input_and_attachment_survive_offline_collection_and_portable_restore(tmp_path: Path) -> None:
    import hashlib
    from pydantic_ai.messages import BinaryContent, TextContent
    from linktools.ai.inputs import ExecutionInputMaterializer
    from linktools.ai.core import PromptLimits
    from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
    from linktools.ai.persistence.objects import ObjectKeyFactory
    from linktools.ai.persistence.snapshots import SnapshotLimits
    from linktools.ai.storage import InMemoryObjectStore, ObjectRef, read_object
    from .test_evaluation_retention import Offline
    from .test_runtime_persistence_evidence import _execution as standalone_execution

    namespace = "payloads-portable"
    configuration = PersistenceStorage.from_root(tmp_path / "source", exclusive_writer=True)
    attachment = b"\x00\xff retained attachment"
    prompt = [TextContent("large user input " * 12000, metadata={
        "opaque": {"$execution_payload": "invalid", "$execution_binding": "invalid",
                   "$dataclass": "archive_root", "archive_scope": "opaque", "content_scope": "opaque"}}),
        BinaryContent(attachment, media_type="application/octet-stream")]
    async with Persistence.open(namespace, storage=configuration, tenant_id="tenant") as state:
        objects = state.object_store(PersistenceDomain.EXECUTION)
        materializer = ExecutionInputMaterializer(None, PromptLimits(), object_store=objects,
            object_key_factory=ObjectKeyFactory(namespace), object_domain=PersistenceDomain.EXECUTION)
        stored = await materializer.store(prompt, tenant_id="tenant")
        execution = replace(standalone_execution(), stored_user_input=stored)
        await state.execution.executions.create_execution(execution)
        refs = stored.payload.object_refs()
        assert len(refs) == 1
        await state.purge_unreferenced_objects(tuple((PersistenceDomain.EXECUTION, reference) for reference in refs), exclusive=Offline())
        assert await read_object(objects, refs[0].key, expected_digest=refs[0].digest, expected_size=refs[0].size) == attachment
        decoded = await materializer.restore((await state.execution.executions.get_fresh("execution", tenant_id="tenant")).stored_user_input)
        assert decoded[0] == prompt[0]
        assert decoded[1].data == attachment and decoded[1].media_type == prompt[1].media_type
        # A candidate is a real external object, not any large ordinary JSON body.
        data = b"orphan"
        async def chunks():
            yield data
        stat = await objects.put("orphan", chunks(), expected_size=len(data), expected_digest=hashlib.sha256(data).hexdigest())
        await state.purge_unreferenced_objects(((PersistenceDomain.EXECUTION, ObjectRef(objects.store_id, stat.key, stat.digest, stat.size)),), exclusive=Offline())
        assert await objects.stat("orphan") is None
    snapshots = InMemoryObjectStore()
    limits = SnapshotLimits(10000, 10 * 1024 * 1024)
    async with Persistence.open(namespace, storage=configuration, tenant_id="tenant", read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=snapshots, limits=limits)
    target = tmp_path / "target"
    await Persistence.restore_snapshot(snapshot, object_store=snapshots, root=target, limits=limits)
    async with Persistence.open(namespace, storage=PersistenceStorage.from_root(target), tenant_id="tenant") as state:
        restored = await state.execution.executions.get_fresh("execution", tenant_id="tenant")
        materializer = ExecutionInputMaterializer(None, PromptLimits(), object_store=state.object_store(PersistenceDomain.EXECUTION),
            object_key_factory=ObjectKeyFactory(namespace), object_domain=PersistenceDomain.EXECUTION)
        decoded = await materializer.restore(restored.stored_user_input)
        assert decoded[0] == prompt[0]
        assert decoded[1].data == attachment and decoded[1].media_type == prompt[1].media_type


@pytest.mark.asyncio
async def test_complete_batch_reads_share_decoded_payloads_without_reencoding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="payloads", tenant_id="tenant")
    try:
        repository = state.execution.executions
        value = _execution("first", status=ExecutionStatus.SUCCEEDED, prompt="shared input " * 16000)
        await repository.create_execution(value)
        await repository.create_execution(replace(value, execution_id="second", root_execution_id="second"))
        import linktools.ai.persistence._execution_content as content_module
        original = content_module._encode_persisted_domain
        def reject_reencoding(item):
            from linktools.ai.model import StoredUserInput
            if isinstance(item, (StoredPayload, StoredUserInput)):
                raise AssertionError("complete list read must inject verified values directly")
            return original(item)
        monkeypatch.setattr(content_module, "_encode_persisted_domain", reject_reencoding)
        values = await repository.get_many_fresh(("first", "second"), tenant_id="tenant")
        assert values["first"].stored_user_input is values["second"].stored_user_input
        assert values["first"].stored_user_input == value.stored_user_input
        listed = await repository.list_by_session(value.session_id, tenant_id="tenant")
        assert len(listed) == 2 and listed[0].stored_user_input is listed[1].stored_user_input
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_revision", (True, "1"))
async def test_control_cas_rejects_malformed_stored_revision(tmp_path: Path, malformed_revision: object) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="payloads", tenant_id="tenant")
    try:
        repository = state.execution.executions
        value = _execution("execution", status=ExecutionStatus.STARTED, prompt="body " * 20000)
        value = replace(value, revision=1)
        await repository.create_execution(value)
        record = await repository.state_store.read_record(repository.execution_record_key(value.execution_id))
        from copy import deepcopy
        data = deepcopy(dict(record.data))
        data["value"]["payload"]["fields"]["revision"] = malformed_revision
        await repository.state_store.mutate(lambda tx: tx.replace_record(replace(record, data=data, storage_revision=record.storage_revision + 1),
            expected_storage_revision=record.storage_revision))
        with pytest.raises(AIError) as invalid:
            await repository.compare_and_swap(value.execution_id, tenant_id="tenant", expected_revision=1,
                next_record=replace(value, revision=2))
        assert invalid.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_offline_collection_retains_execution_payloads_in_other_physical_store_scopes(tmp_path: Path) -> None:
    from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
    from linktools.ai.persistence import PersistenceDomain
    from .test_evaluation_retention import Offline
    from .test_runtime_persistence_evidence import _execution as standalone_execution
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="caller", tenant_id="tenant")
    try:
        repository = ExecutionRepositoryImpl(state.execution.executions.state_store, namespace="other", tenant_id="other-tenant")
        await repository.initialize_content(create=True)
        value = replace(standalone_execution(),
            stored_user_input=replace(standalone_execution().stored_user_input, payload=StoredPayload.inline_json("other body " * 20000)))
        await repository.create_execution(value)
        from .test_object_cleanup_io import _put_attachment
        objects = state.object_store(PersistenceDomain.EXECUTION)
        candidate = await _put_attachment(objects, "orphan")
        result = await state.purge_unreferenced_objects(((PersistenceDomain.EXECUTION, candidate),), exclusive=Offline())
        assert result.deleted == ((PersistenceDomain.EXECUTION, candidate),)
        assert await repository.get_fresh(value.execution_id, tenant_id="other-tenant") == value
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_terminal_cas_accepts_declared_default_omission_with_content_refs(tmp_path: Path) -> None:
    state = Persistence(_storage("memory", tmp_path))
    await state.initialize(namespace="payloads", tenant_id="tenant")
    try:
        repository = state.execution.executions
        value = _execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="body " * 20000)
        await repository.create_execution(value)
        record = await repository.state_store.read_record(repository.execution_record_key(value.execution_id))
        from copy import deepcopy
        data = deepcopy(dict(record.data))
        del data["value"]["payload"]["fields"]["conversation"]
        await repository.state_store.mutate(lambda tx: tx.replace_record(replace(record, data=data, storage_revision=record.storage_revision + 1),
            expected_storage_revision=record.storage_revision))
        changed = replace(value, revision=value.revision + 1, retention_closed=True)
        assert await repository.compare_and_swap(value.execution_id, tenant_id="tenant", expected_revision=value.revision, next_record=changed) == changed
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_body", (False, True))
async def test_cold_content_snapshot_discovers_its_dependency_once_and_fails_closed(
    tmp_path: Path, missing_body: bool,
) -> None:
    from linktools.ai.persistence import PersistenceStorage
    configuration = PersistenceStorage.sqlite(tmp_path / "cold-content.db")
    value = _execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="cold body " * 12000)
    async with Persistence.open("payloads", storage=configuration, tenant_id="tenant") as state:
        await state.execution.executions.create_execution(value)
        if missing_body:
            _, archive = state.execution.executions.state_store.storage_group.opened_archive
            rows = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
            for row in rows:
                await archive.mutate(lambda tx: tx.delete_record(row.key))
    async with Persistence.open("payloads", storage=configuration, tenant_id="tenant", read_only=True) as state:
        repository = state.execution.executions
        group = repository.state_store.storage_group
        assert group.opened_archive is None
        captures = 0
        async def capture(_group):
            nonlocal captures
            captures += 1
            return await repository.get_timeline_many((value.execution_id,), tenant_id="tenant")
        if missing_body:
            with pytest.raises(AIError) as missing:
                await repository.read_content(capture)
            assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            records = await repository.read_content(capture)
            assert records[value.execution_id].stored_user_input == value.stored_user_input
            assert records[value.execution_id].result == value.result
        assert captures == 2


@pytest.mark.asyncio
async def test_cold_content_snapshot_restarts_after_another_reader_opens_its_archive(tmp_path: Path) -> None:
    import asyncio
    from linktools.ai.persistence import PersistenceStorage
    configuration = PersistenceStorage.sqlite(tmp_path / "concurrent-cold-content.db")
    value = _execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="concurrent cold body " * 8000)
    async with Persistence.open("payloads", storage=configuration, tenant_id="tenant") as state:
        await state.execution.executions.create_execution(value)
    async with Persistence.open("payloads", storage=configuration, tenant_id="tenant", read_only=True) as state:
        repository = state.execution.executions
        group = repository.state_store.storage_group
        assert group.opened_archive is None
        entered, opened = asyncio.Event(), asyncio.Event()
        async def open_from_another_reader():
            await entered.wait()
            await group.open_archive()
            opened.set()
        opener = asyncio.create_task(open_from_another_reader())
        captures = 0
        async def capture(_group):
            nonlocal captures
            captures += 1
            if captures == 1:
                entered.set()
                await asyncio.wait_for(opened.wait(), timeout=10)
            return await repository.get_timeline_many((value.execution_id,), tenant_id="tenant")
        try:
            records = await repository.read_content(capture)
        finally:
            await opener
        assert captures == 2
        assert records[value.execution_id].stored_user_input == value.stored_user_input
        assert records[value.execution_id].result == value.result


@pytest.mark.asyncio
@pytest.mark.parametrize("slot", ("stored_user_input", "output"))
@pytest.mark.parametrize("schema", (999, None, True))
async def test_execution_reference_wrapper_damage_is_rejected_before_hydration(
    tmp_path: Path, slot: str, schema: object,
) -> None:
    from copy import deepcopy

    async with Persistence.open("payloads", tenant_id="tenant", storage=_storage("memory", tmp_path)) as state:
        repository = state.execution.executions
        value = _execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="input " * 20000)
        value = replace(value, result=replace(value.result, output=StoredPayload.inline_json("output " * 20000)))
        await repository.create_execution(value)
        record = await repository.state_store.read_record(repository.execution_record_key(value.execution_id))
        scope, archive = repository.state_store.storage_group.opened_archive
        rows = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        contents = {(scope, archive_scope_token(row.scope_key), row.sort_key.decode("utf-8")):
                    ContentRecord.from_record(row) for row in rows}
        data = deepcopy(dict(record.data))
        fields = data["value"]["payload"]["fields"]
        wrapper = fields["stored_user_input"] if slot == "stored_user_input" else fields["result"]["fields"]["output"]
        assert "content" in wrapper["fields"]
        if schema is None:
            del wrapper["schema"]
        else:
            wrapper["schema"] = schema
        damaged = replace(record, data=data)
        expected = ErrorCode.STORAGE_VERSION_UNSUPPORTED if schema == 999 else ErrorCode.STORAGE_INTEGRITY_ERROR
        # Supply corrupted stored bytes directly: public writes correctly reject
        # these wrappers before they can be installed in the database.
        with pytest.raises(AIError) as runtime_error:
            await repository._execution_content.decode(damaged)
        assert runtime_error.value.code is expected
        with pytest.raises(AIError) as offline_error:
            expand_snapshot_execution_content(damaged, contents, namespace="payloads", tenant_id="tenant")
        assert offline_error.value.code is expected
