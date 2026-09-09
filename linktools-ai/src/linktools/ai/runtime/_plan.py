#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned planning command and durable plan projection."""

import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from linktools.core import environ

from ..core import (
    JsonValue,
    OperationKind,
    OperationLedgerInput,
    OperationLedgerRecord,
    OperationStatus,
    ResourceKind,
    canonical_sha256,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from .state import (
    StateStore,
    StateTransaction,
    StoredRecord,
)
from .state._contracts import (
    OperationLedgerRepository,
)
from .state import partition_digest, record_key_digest

_logger = environ.get_logger("ai.runtime.plan")
_OWNER_KINDS = frozenset({"session", "execution"})
_KIND = "agent_plan"
_VERSION = 2
_MAX_ITEMS = 128
_MAX_BYTES = 64 * 1024
_SHA256 = re.compile(r"[0-9a-f]{64}")

PlanOwnerKind = Literal["session", "execution"]
PlanStatus = Literal["pending", "in_progress", "completed", "cancelled"]


@dataclass(frozen=True, slots=True)
class PlanItem:
    content: str
    status: PlanStatus = "pending"


@dataclass(frozen=True, slots=True)
class PlanOperation:
    id: str
    fingerprint: str


class RuntimePlanStore:
    """Persist one complete plan through the existing optimistic state store."""

    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
        owner_kind: PlanOwnerKind,
        owner_id: str,
        operations: OperationLedgerRepository | None = None,
    ) -> None:
        if (
            not isinstance(owner_kind, str)
            or owner_kind not in _OWNER_KINDS
            or not isinstance(owner_id, str)
            or not owner_id
        ):
            raise ValueError("plan owner is invalid")
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id
        validate_tenant_id(tenant_id)
        self._owner_kind = owner_kind
        self._owner_id = owner_id
        domain = "conversation" if owner_kind == "session" else "execution"
        self._key = record_key_digest(
            namespace,
            tenant_id,
            domain,
            _KIND,
            [owner_kind, owner_id],
        )
        self._partition = partition_digest(namespace, tenant_id, domain, _KIND)
        self._operations = operations

    @property
    def owner_kind(self) -> PlanOwnerKind:
        return self._owner_kind

    @property
    def owner_id(self) -> str:
        return self._owner_id

    async def get_items(self) -> list[PlanItem]:
        record = await self._store.read(
            lambda transaction: transaction.get_record(self._key)
        )
        return _decode_payload(
            record,
            owner_kind=self._owner_kind,
            owner_id=self._owner_id,
        )[0]

    async def get_plan(self) -> dict[str, JsonValue]:
        record = await self._store.read(
            lambda transaction: transaction.get_record(self._key)
        )
        items, revision = _decode_payload(
            record,
            owner_kind=self._owner_kind,
            owner_id=self._owner_id,
        )
        return {
            "items": [_item_payload(item) for item in items],
            "revision": revision,
        }

    async def write_plan(
        self,
        items: list[PlanItem],
        *,
        operation: PlanOperation | None = None,
    ) -> dict[str, JsonValue]:
        if self._operations is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        values = _validated_items(items)
        operation = _ensure_operation(operation, values)
        operation_id = _operation_id(
            operation,
            self._owner_kind,
            self._owner_id,
        )
        replay = await self._replay(operation_id, operation.fingerprint)
        if replay is not None:
            return replay

        async def mutate(transaction: StateTransaction) -> dict[str, JsonValue]:
            current = await transaction.get_record(self._key)
            existing = await self._operations.get_in_transaction(
                transaction,
                operation_id,
                tenant_id=self._tenant_id,
            )
            if existing is not None:
                if (
                    not _plan_operation_matches(
                        existing,
                        self._owner_kind,
                        self._owner_id,
                    )
                    or existing.request_digest != operation.fingerprint
                    or existing.status is not OperationStatus.SUCCEEDED
                    or existing.result_ref is None
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                return _decode_receipt(
                    existing.result_ref,
                    expected_fingerprint=operation.fingerprint,
                )
            current_revision = _decode_payload(
                current,
                owner_kind=self._owner_kind,
                owner_id=self._owner_id,
            )[1]
            revision = 1 if current is None else current_revision + 1
            next_record = self._record(values, revision, current)
            if current is None:
                await transaction.insert_record(next_record)
            elif not await transaction.replace_record(
                next_record,
                expected_storage_version=current.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            result = {
                "items": [_item_payload(item) for item in values],
                "revision": revision,
            }
            await self._operations.append_in_transaction(
                transaction,
                _operation_input(
                    operation,
                    self._tenant_id,
                    self._owner_kind,
                    self._owner_id,
                    result,
                ),
            )
            _logger.debug(
                "runtime plan replaced: owner_kind=%s owner_id=%s revision=%s items=%s",
                self._owner_kind,
                self._owner_id,
                revision,
                len(values),
            )
            return {
                "items": [_item_payload(item) for item in values],
                "revision": revision,
            }

        return await self._store.mutate(mutate)

    async def _replay(
        self,
        operation_id: str,
        fingerprint: str,
    ) -> dict[str, JsonValue] | None:
        if self._operations is None:
            return None
        current = await self._operations.get(
            operation_id,
            tenant_id=self._tenant_id,
        )
        if current is None:
            return None
        if (
            not _plan_operation_matches(current, self._owner_kind, self._owner_id)
            or current.request_digest != fingerprint
            or current.status is not OperationStatus.SUCCEEDED
            or current.result_ref is None
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return _decode_receipt(
            current.result_ref,
            expected_fingerprint=fingerprint,
        )

    def _record(
        self,
        items: list[PlanItem],
        revision: int,
        current: StoredRecord | None,
    ) -> StoredRecord:
        payload: dict[str, JsonValue] = {
            "version": _VERSION,
            "owner_kind": self._owner_kind,
            "owner_id": self._owner_id,
            "revision": revision,
            "items": [_item_payload(item) for item in items],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > _MAX_BYTES:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return StoredRecord(
            key_digest=self._key,
            partition_digest=self._partition,
            scope_digest=None,
            parent_digest=None,
            kind=_KIND,
            sort_key="plan:" + self._key.hex(),
            state=None,
            storage_version=1 if current is None else current.storage_version + 1,
            lease_owner=None,
            lease_fence=0,
            lease_expires_at=None,
            data=payload,
        )


def _decode_payload(
    record: StoredRecord | None,
    *,
    owner_kind: PlanOwnerKind,
    owner_id: str,
) -> tuple[list[PlanItem], int]:
    if record is None:
        return [], 0
    if (
        record.kind != _KIND
        or isinstance(record.storage_version, bool)
        or not isinstance(record.storage_version, int)
        or record.storage_version < 1
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    data = record.data
    if not isinstance(data, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = data.get("version")
    if (
        "version" not in data
        or isinstance(version, bool)
        or not isinstance(version, int)
        or version < 1
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != _VERSION:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if set(data) != {"version", "owner_kind", "owner_id", "revision", "items"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if data.get("owner_kind") != owner_kind:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if data.get("owner_id") != owner_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    revision = data.get("revision")
    raw_items = data.get("items")
    if (
        isinstance(revision, bool)
        or not isinstance(revision, int)
        or revision < 1
        or not isinstance(raw_items, list)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if revision != record.storage_version:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        decoded_items: list[PlanItem] = []
        for item in raw_items:
            if not isinstance(item, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if set(item) != {"content", "status"}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            decoded_items.append(PlanItem(item["content"], item["status"]))
        items = _validated_items(decoded_items)
    except (KeyError, TypeError, ValueError, AIError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(items) != len(raw_items):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return items, revision


def _validated_items(items: list[PlanItem]) -> list[PlanItem]:
    if not isinstance(items, list) or len(items) > _MAX_ITEMS:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    values: list[PlanItem] = []
    for item in items:
        if not isinstance(item, PlanItem):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(item.content, str) or not isinstance(item.status, str):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if item.status not in {
            "pending",
            "in_progress",
            "completed",
            "cancelled",
        }:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not item.content.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        values.append(PlanItem(item.content, item.status))
    encoded = json.dumps(
        [_item_payload(item) for item in values],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > _MAX_BYTES:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return values


def _ensure_operation(
    operation: PlanOperation | None,
    items: list[PlanItem],
) -> PlanOperation:
    if operation is not None:
        if (
            not isinstance(operation, PlanOperation)
            or not isinstance(operation.id, str)
            or not operation.id
            or not isinstance(operation.fingerprint, str)
            or _SHA256.fullmatch(operation.fingerprint) is None
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if operation.fingerprint != _plan_fingerprint(items):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return operation
    return PlanOperation(
        uuid.uuid4().hex,
        _plan_fingerprint(items),
    )


def _plan_fingerprint(items: list[PlanItem]) -> str:
    return canonical_sha256({"items": [_item_payload(item) for item in items]})


def _operation_input(
    operation: PlanOperation,
    tenant_id: str,
    owner_kind: PlanOwnerKind,
    owner_id: str,
    result: dict[str, JsonValue],
) -> OperationLedgerInput:
    now = datetime.now(timezone.utc)
    resource_kind = (
        ResourceKind.SESSION if owner_kind == "session" else ResourceKind.EXECUTION
    )
    return OperationLedgerInput(
        _operation_id(operation, owner_kind, owner_id),
        tenant_id,
        resource_kind,
        owner_id,
        owner_id if owner_kind == "execution" else None,
        OperationKind.TOOL,
        OperationStatus.SUCCEEDED,
        operation.fingerprint,
        json.dumps(
            {"version": _VERSION, "result": result},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        None,
        None,
        True,
        now,
        now,
    )


def _decode_receipt(
    value: str,
    *,
    expected_fingerprint: str | None = None,
) -> dict[str, JsonValue]:
    try:
        raw = json.loads(value)
        if not isinstance(raw, dict):
            raise ValueError("plan receipt is not an object")
        version = raw.get("version")
        if (
            "version" not in raw
            or isinstance(version, bool)
            or not isinstance(version, int)
            or version < 1
        ):
            raise ValueError("plan receipt version is malformed")
        if version != _VERSION:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        result = raw["result"]
        if (
            set(raw) != {"version", "result"}
            or not isinstance(result, dict)
            or set(result) != {"items", "revision"}
            or not isinstance(result["items"], list)
            or isinstance(result["revision"], bool)
            or not isinstance(result["revision"], int)
            or result["revision"] < 1
        ):
            raise ValueError("plan receipt is invalid")
        items = []
        for item in result["items"]:
            if not isinstance(item, Mapping):
                raise ValueError("plan receipt items are invalid")
            if set(item) != {"content", "status"}:
                raise ValueError("plan receipt item fields are invalid")
            items.append(PlanItem(item["content"], item["status"]))
        try:
            _validated_items(items)
        except AIError as error:
            raise ValueError("plan receipt items are invalid") from error
        if (
            expected_fingerprint is not None
            and _plan_fingerprint(items) != expected_fingerprint
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return {
            "items": [_item_payload(item) for item in items],
            "revision": result["revision"],
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _operation_id(
    operation: PlanOperation,
    owner_kind: PlanOwnerKind,
    owner_id: str,
) -> str:
    return canonical_sha256(
        {
            "owner_kind": owner_kind,
            "owner_id": owner_id,
            "call": operation.id,
        }
    )


def _plan_operation_matches(
    operation: OperationLedgerRecord,
    owner_kind: PlanOwnerKind,
    owner_id: str,
) -> bool:
    expected_resource = (
        ResourceKind.SESSION if owner_kind == "session" else ResourceKind.EXECUTION
    )
    expected_execution = owner_id if owner_kind == "execution" else None
    return (
        operation.resource_kind is expected_resource
        and operation.resource_id == owner_id
        and operation.operation_kind is OperationKind.TOOL
        and operation.execution_id == expected_execution
    )


def _item_payload(item: PlanItem) -> dict[str, JsonValue]:
    return {"content": item.content, "status": item.status}


__all__ = [
    "PlanItem",
    "PlanOperation",
    "PlanOwnerKind",
    "PlanStatus",
    "RuntimePlanStore",
]
