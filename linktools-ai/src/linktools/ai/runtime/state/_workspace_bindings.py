#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable trusted path bindings for workspace tool calls."""

import hashlib
from typing import cast

from ...core import JsonValue, canonical_json_bytes
from ...errors import AIError, ErrorCode
from ._codec import decode_domain, decode_envelope, encode_domain, encode_envelope
from ._contracts import WorkspaceToolCallBinding
from ._plan import RuntimeDomain
from ._store import (
    FactQuery,
    StateStore,
    StateTransaction,
    StoredFact,
    StoredRecord,
    partition_digest,
    record_key_digest,
    sequence_key,
    sortable_id,
    stream_digest,
)

_BINDING_KIND = "workspace_tool_binding"
_BINDING_RELATION = "workspace-tool-call-binding"
_BINDING_VERSION = 1


def workspace_tool_call_binding_subject_digest(
    namespace: str,
    tenant_id: str,
    step_run_id: str,
    tool_call_id: str,
) -> bytes:
    """Return the exact subject identity for one trusted tool call."""
    return hashlib.sha256(
        canonical_json_bytes(
            [
                "workspace-tool-call-binding-v1",
                namespace,
                tenant_id,
                step_run_id,
                tool_call_id,
            ]
        )
    ).digest()


class WorkspaceToolCallBindingStore:
    """Persist trusted path bindings for the lifetime of one execution."""

    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
    ) -> None:
        if not namespace or not tenant_id:
            raise ValueError("namespace and tenant_id are required")
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id

    def _owner(self, execution_id: str) -> bytes:
        return record_key_digest(
            self._namespace,
            self._tenant_id,
            RuntimeDomain.RECOVERY.value,
            _BINDING_RELATION,
            ["v1", execution_id],
        )

    def _stream(self, execution_id: str) -> bytes:
        return stream_digest(
            self._namespace,
            self._tenant_id,
            RuntimeDomain.RECOVERY.value,
            _BINDING_RELATION,
            ["v1", execution_id],
        )

    def _sequence(self, execution_id: str) -> bytes:
        return sequence_key(
            self._namespace,
            self._tenant_id,
            RuntimeDomain.RECOVERY.value,
            _BINDING_RELATION,
            ["v1", execution_id],
        )

    async def store(
        self,
        binding: WorkspaceToolCallBinding,
    ) -> WorkspaceToolCallBinding:
        if not isinstance(binding, WorkspaceToolCallBinding):
            raise TypeError("binding must be WorkspaceToolCallBinding")
        subject = workspace_tool_call_binding_subject_digest(
            self._namespace,
            self._tenant_id,
            binding.step_run_id,
            binding.tool_call_id,
        )
        encoded = _encode_binding(binding)
        stream = self._stream(binding.execution_id)
        owner_key = self._owner(binding.execution_id)
        sequence_key_value = self._sequence(binding.execution_id)

        async def mutate(transaction: StateTransaction) -> WorkspaceToolCallBinding:
            owner = await _ensure_owner(transaction, self, binding.execution_id)
            facts = await transaction.list_facts(
                FactQuery(stream, subject_digest=subject, latest=True)
            )
            if facts:
                existing = _decode_binding(facts[0])
                if existing != binding:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return existing
            if await transaction.guard_record(
                owner_key,
                expected_storage_version=owner.storage_version,
            ) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            sequence = await transaction.reserve_sequence(sequence_key_value, 1)
            await transaction.insert_fact(
                StoredFact(
                    stream,
                    sequence,
                    owner_key,
                    _BINDING_KIND,
                    subject,
                    binding.error_code,
                    encoded,
                )
            )
            return binding

        stored = await self._store.mutate(mutate)
        readback = await self.get(
            binding.execution_id,
            binding.step_run_id,
            binding.tool_call_id,
        )
        if readback != stored:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return readback

    async def get(
        self,
        execution_id: str,
        step_run_id: str,
        tool_call_id: str,
    ) -> WorkspaceToolCallBinding | None:
        subject = workspace_tool_call_binding_subject_digest(
            self._namespace,
            self._tenant_id,
            step_run_id,
            tool_call_id,
        )
        stream = self._stream(execution_id)

        async def read(transaction: StateTransaction) -> WorkspaceToolCallBinding | None:
            facts = await transaction.list_facts(
                FactQuery(stream, subject_digest=subject, latest=True)
            )
            if not facts:
                return None
            if len(facts) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            binding = _decode_binding(facts[0])
            if (
                binding.execution_id != execution_id
                or binding.step_run_id != step_run_id
                or binding.tool_call_id != tool_call_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return binding

        return await self._store.read(read)

    async def release_execution(self, execution_id: str) -> None:
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("execution_id is required")
        owner_key = self._owner(execution_id)
        sequence_key_value = self._sequence(execution_id)

        async def mutate(transaction: StateTransaction) -> None:
            owner = await transaction.get_record(owner_key)
            if owner is None:
                return
            _validate_owner(owner)
            await transaction.delete_fact_streams(owner_key)
            await transaction.delete_sequence(sequence_key_value)
            if not await transaction.delete_record(
                owner_key,
                expected_storage_version=owner.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)

        await self._store.mutate(mutate)


async def _ensure_owner(
    transaction: StateTransaction,
    store: WorkspaceToolCallBindingStore,
    execution_id: str,
) -> StoredRecord:
    owner_key = store._owner(execution_id)
    owner = await transaction.get_record(owner_key)
    if owner is None:
        owner = StoredRecord(
            owner_key,
            partition_digest(
                store._namespace,
                store._tenant_id,
                RuntimeDomain.RECOVERY.value,
                _BINDING_RELATION,
            ),
            None,
            None,
            _BINDING_KIND,
            sortable_id(execution_id),
            "active",
            0,
            None,
            0,
            None,
            {"version": _BINDING_VERSION},
        )
        await transaction.insert_record(owner)
        return owner
    _validate_owner(owner)
    return owner


def _validate_owner(owner: StoredRecord) -> None:
    if owner.kind != _BINDING_KIND or owner.data != {"version": _BINDING_VERSION}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _encode_binding(binding: WorkspaceToolCallBinding) -> dict[str, JsonValue]:
    return encode_envelope(
        {
            "type": "workspace_tool_call_binding",
            "payload": encode_domain(binding),
        }
    )


def _decode_binding(fact: StoredFact) -> WorkspaceToolCallBinding:
    if fact.kind != _BINDING_KIND:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        envelope = decode_envelope(fact.data)
        if set(envelope.value) != {"type", "payload"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if envelope.value["type"] != "workspace_tool_call_binding":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cast(
            WorkspaceToolCallBinding,
            decode_domain(
                envelope.value["payload"],
                WorkspaceToolCallBinding,
            ),
        )
    except AIError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


__all__ = [
    "WorkspaceToolCallBindingStore",
    "workspace_tool_call_binding_subject_digest",
]
