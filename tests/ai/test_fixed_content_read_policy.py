#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed content addresses locate bytes; typed decoding still validates structure."""
import hashlib
import json
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import ContentRecord
from linktools.ai.persistence._execution_content import expand_snapshot_execution_content
from linktools.ai.persistence._archive import archive_scope_token
from linktools.ai.persistence._payload import decode_payload_bytes
from linktools.ai.persistence.codec import _decode_enveloped_domain
from linktools.ai.persistence.contracts import ExecutionRecord
from linktools.ai.persistence.state import RecordQuery

from .test_incremental_projection_boundaries import _storage
from .test_session_timeline import _execution


@pytest.mark.parametrize("compression", ("raw", "zlib"))
def test_fixed_payload_decoding_preserves_limits_not_byte_digest(compression: str) -> None:
    raw = b'{"valid":"new"}'
    encoded = zlib.compress(raw) if compression == "zlib" else raw
    assert decode_payload_bytes(encoded, codec=compression, raw_digest="0" * 64, raw_size=len(raw)) == raw
    with pytest.raises(AIError):
        decode_payload_bytes(encoded, codec=compression, raw_digest="0" * 64, raw_size=len(raw) - 1)
    with pytest.raises(AIError):
        decode_payload_bytes(encoded, codec="unsupported", raw_digest="0" * 64, raw_size=len(raw))


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_execution_fixed_owners_accept_changed_bytes_but_reject_invalid_structure(
    tmp_path: Path, backend: str,
) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace="fixed-read", tenant_id="tenant")
    try:
        value = _execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="input")
        repository = state.execution.executions
        await repository.create_execution(value)
        owner = await repository.state_store.read_record(repository.execution_record_key("execution"))
        scope, archive = repository.state_store.storage_group.opened_archive
        rows = await archive.read(lambda tx: tx.list_records(RecordQuery(kind="archive_content")))
        contents = {}
        for row in rows:
            original = ContentRecord.from_record(row)
            raw = decode_payload_bytes(original.body, codec=original.compression,
                                       raw_digest=original.raw_digest, raw_size=original.raw_size)
            parsed = json.loads(raw)
            changed = json.dumps(dict(reversed(tuple(parsed.items()))), ensure_ascii=False, separators=(",", ":")).encode()
            assert len(changed) == len(raw) and changed != raw
            assert hashlib.sha256(changed).hexdigest() != original.raw_digest
            updated = replace(row, storage_revision=row.storage_revision + 1, payload_bytes=changed,
                              data={**row.data, "compression": "raw"})
            assert await archive.mutate(lambda tx: tx.replace_record(updated, expected_storage_revision=row.storage_revision))
            contents[(scope, archive_scope_token(row.scope_key), row.sort_key.decode())] = replace(
                original, compression="raw", body=changed)
        assert await repository.get_fresh("execution", tenant_id="tenant") == value
        restored = expand_snapshot_execution_content(owner, contents, namespace="fixed-read", tenant_id="tenant")
        assert _decode_enveloped_domain(restored.data, ExecutionRecord) == value
        for reference, content in contents.items():
            malformed = {**contents, reference: replace(content, body=b"[" * content.raw_size)}
            with pytest.raises(AIError):
                expand_snapshot_execution_content(owner, malformed, namespace="fixed-read", tenant_id="tenant")
        with pytest.raises(AIError):
            expand_snapshot_execution_content(owner, {}, namespace="fixed-read", tenant_id="tenant")
    finally:
        await state.close()
