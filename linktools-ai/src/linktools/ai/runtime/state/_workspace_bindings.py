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
    """Persist and read trusted workspace path bindings as StateStore facts."""

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
        self._stream = stream_digest(
            namespace,
            tenant_id,
            RuntimeDomain.RECOVERY.value,
            _BINDING_RELATION,
            _BINDING_VERSION,
        )
        self._owner = record_key_digest(
            namespace,
            tenant_id,
            RuntimeDomain.RECOVERY.value,
            _BINDING_RELATION,
            _BINDING_VERSION,
        )
        self._sequence = sequence_key(
            namespace,
            tenant_id,
            RuntimeDomain.RECOVERY.value,
            _BINDING_RELATION,
            _BINDING_VERSION,
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

        async def mutate(transaction: StateTransaction) -> WorkspaceToolCallBinding:
            owner = await _ensure_owner(transaction, self)
            facts = await transaction.list_facts(
                FactQuery(self._stream, subject_digest=subject, latest=True)
            )
            if facts:
                existing = _decode_binding(facts[0])
                if existing != binding:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return existing
            if await transaction.guard_record(
                self._owner,
                expected_storage_version=owner.storage_version,
            ) is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            sequence = await transaction.reserve_sequence(self._sequence, 1)
            await transaction.insert_fact(
                StoredFact(
                    self._stream,
                    sequence,
                    self._owner,
                    _BINDING_KIND,
                    subject,
                    binding.error_code,
                    encoded,
                )
            )
            return binding

        stored = await self._store.mutate(mutate)
        readback = await self.get(binding.step_run_id, binding.tool_call_id)
        if readback != stored:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return readback

    async def get(
        self,
        step_run_id: str,
        tool_call_id: str,
    ) -> WorkspaceToolCallBinding | None:
        subject = workspace_tool_call_binding_subject_digest(
            self._namespace,
            self._tenant_id,
            step_run_id,
            tool_call_id,
        )

        async def read(transaction: StateTransaction) -> WorkspaceToolCallBinding | None:
            facts = await transaction.list_facts(
                FactQuery(self._stream, subject_digest=subject, latest=True)
            )
            if not facts:
                return None
            if len(facts) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            binding = _decode_binding(facts[0])
            if (
                binding.step_run_id != step_run_id
                or binding.tool_call_id != tool_call_id
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return binding

        return await self._store.read(read)


async def _ensure_owner(
    transaction: StateTransaction,
    store: WorkspaceToolCallBindingStore,
) -> StoredRecord:
    owner = await transaction.get_record(store._owner)
    if owner is None:
        owner = StoredRecord(
            store._owner,
            partition_digest(
                store._namespace,
                store._tenant_id,
                RuntimeDomain.RECOVERY.value,
                _BINDING_RELATION,
            ),
            None,
            None,
            _BINDING_KIND,
            sortable_id("v1"),
            "active",
            0,
            None,
            0,
            None,
            {"version": _BINDING_VERSION},
        )
        await transaction.insert_record(owner)
        return owner
    if owner.kind != _BINDING_KIND or owner.data != {"version": _BINDING_VERSION}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return owner


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
