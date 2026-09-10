#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned planning command and durable plan projection."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from linktools.core import environ

from ..core import (
    JsonValue,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from .state._store import (
    StateStore,
    StateTransaction,
    StoredRecord,
    partition_digest,
    record_key_digest,
)

_logger = environ.get_logger("ai.runtime.plan")
_OWNER_KINDS = frozenset({"session", "execution"})
_KIND = "agent_plan"
_VERSION = 1

PlanOwnerKind = Literal["session", "execution"]
PlanStatus = Literal["pending", "in_progress", "completed", "cancelled"]


@dataclass(frozen=True, slots=True)
class PlanItem:
    content: str
    status: PlanStatus = "pending"


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
    ) -> dict[str, JsonValue]:
        values = _validated_items(items)

        async def mutate(transaction: StateTransaction) -> dict[str, JsonValue]:
            current = await transaction.get_record(self._key)
            current_items, current_revision = _decode_payload(
                current, owner_kind=self._owner_kind, owner_id=self._owner_id
            )
            if current_items == values:
                return {
                    "items": [_item_payload(item) for item in values],
                    "revision": current_revision,
                }
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
            _logger.debug(
                "runtime plan replaced: owner_kind=%s owner_id=%s revision=%s items=%s",
                self._owner_kind,
                self._owner_id,
                revision,
                len(values),
            )
            return result

        try:
            return await self._store.mutate(mutate)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            current = await self.get_items()
            if current == values:
                return {
                    "items": [_item_payload(item) for item in values],
                    "revision": (await self.get_plan())["revision"],
                }
            raise

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
    if not {
        "version",
        "owner_kind",
        "owner_id",
        "revision",
        "items",
    }.issubset(data):
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
            if not {"content", "status"}.issubset(item):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            decoded_items.append(PlanItem(item["content"], item["status"]))
        items = _validated_items(decoded_items)
    except (KeyError, TypeError, ValueError, AIError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(items) != len(raw_items):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return items, revision


def _validated_items(items: list[PlanItem]) -> list[PlanItem]:
    if not isinstance(items, list):
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
    return values


def _item_payload(item: PlanItem) -> dict[str, JsonValue]:
    return {"content": item.content, "status": item.status}


__all__ = [
    "PlanItem",
    "PlanOwnerKind",
    "PlanStatus",
    "RuntimePlanStore",
]
