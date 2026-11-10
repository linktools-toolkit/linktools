#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate portable physical Trace references against their native owner backup."""

import json
from collections.abc import Mapping
from dataclasses import replace

from pydantic import TypeAdapter
from pydantic_ai_harness.step_persistence import RunRecord

from ..errors import AIError, ErrorCode
from ._native_run_reference import NativeRunReferences, is_native_run_reference
from ._native_event_reference import NativeEventReferences
from ._step_backup import NativeStepBackup
from ._step_store import SqlStepStore
from .codec import _encode_step_envelope
from .state import StoredRecord, StoredFact, record_key_digest


_RUN_ADAPTER = TypeAdapter(RunRecord)


class NativeBackupProjections:
    """One validated inventory index shared by all portable Trace references."""

    def __init__(self, backup: NativeStepBackup, *, namespace: str, tenant_id: str) -> None:
        self._namespace, self._tenant_id = namespace, tenant_id
        self._store_digest = SqlStepStore.scope_digest(namespace=namespace, tenant_id=tenant_id)
        try:
            if backup.store_digest != self._store_digest:
                raise ValueError("native backup scope differs from its Trace partition")
            SqlStepStore.validate_backup(backup)
        except (TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        self._runs = {row["run_id"]: row for row in backup.runs}
        self._events = {row["key_digest"]: row for row in backup.events}

    def validate(self, projected: StoredRecord | StoredFact,
                 physical_data: Mapping[str, object]) -> StoredRecord | StoredFact:
        """Return the physical locator only when its full projection is unchanged."""
        try:
            physical = replace(projected, data=physical_data)
            if isinstance(projected, StoredRecord):
                if projected.kind != "agent_run" or not is_native_run_reference(physical_data):
                    raise ValueError("portable native run projection has an invalid kind")
                reference = NativeRunReferences.decode_reference(physical_data)
                row = self._runs[reference["run"]]
                source = _RUN_ADAPTER.validate_python(dict(row, metadata=json.loads(row["metadata"])))
                run = NativeRunReferences.project_source(physical_data, source,
                    store_digest=row["store_digest"], expected_store_digest=self._store_digest)
                if projected.key_digest != record_key_digest(
                        self._namespace, self._tenant_id, "trace", "agent_run", source.run_id):
                    raise ValueError("portable run projection belongs to another Trace record")
                restored = replace(physical, data=_encode_step_envelope(run))
            elif isinstance(projected, StoredFact):
                reference = NativeEventReferences.decode_reference(physical_data)
                row = self._events[reference["key"]]
                restored = NativeEventReferences.project_source(physical, row,
                    run=self._runs.get(row["run_id"]), namespace=self._namespace, tenant_id=self._tenant_id)
            else:
                raise TypeError("portable native projection must be a record or fact")
            if restored != projected:
                raise ValueError("portable native projection differs from its owner")
            return physical
        except (TypeError, ValueError, KeyError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def validate_native_backup_projection(
    backup: NativeStepBackup, projected: StoredRecord | StoredFact, physical_data: Mapping[str, object],
    *, namespace: str, tenant_id: str,
) -> StoredRecord | StoredFact:
    """Validate one physical projection; reuse NativeBackupProjections for batches."""
    return NativeBackupProjections(backup, namespace=namespace, tenant_id=tenant_id).validate(projected, physical_data)


__all__ = ["NativeBackupProjections", "validate_native_backup_projection"]
