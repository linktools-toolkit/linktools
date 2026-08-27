#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeState-backed Harness planning store."""

from typing import Literal, cast

from pydantic_ai_harness.planning import PlanItem, TaskStatus

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from .state import StateStore, StateTransaction, StoredRecord, partition_digest, record_key_digest

_OWNER_KIND = Literal["session", "execution"]
_KIND = "agent_plan"


class RuntimePlanStore:
    """Persist one owner plan through the existing optimistic StateStore contract."""

    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
        owner_kind: _OWNER_KIND,
        owner_id: str,
    ) -> None:
        if owner_kind not in {"session", "execution"} or not owner_id:
            raise ValueError("plan owner is invalid")
        self._store = store
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._owner_kind = owner_kind
        self._owner_id = owner_id
        self._key = record_key_digest(
            namespace,
            tenant_id,
            "conversation" if owner_kind == "session" else "execution",
            _KIND,
            [owner_kind, owner_id],
        )
        self._partition = partition_digest(
            namespace,
            tenant_id,
            "conversation" if owner_kind == "session" else "execution",
            _KIND,
        )

    async def get_items(self) -> list[PlanItem]:
        record = await self._store.read(lambda transaction: transaction.get_record(self._key))
        return _decode_items(record)

    async def set_items(self, items: list[PlanItem]) -> None:
        values = _validated_items(items)

        async def mutate(transaction: StateTransaction) -> None:
            current = await transaction.get_record(self._key)
            next_record = self._record(values, current)
            if current is None:
                await transaction.insert_record(next_record)
            elif not await transaction.replace_record(
                next_record,
                expected_storage_version=current.storage_version,
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)

        await self._store.mutate(mutate)

    async def get_item(self, item_id: str) -> "PlanItem | None":
        return next((item for item in await self.get_items() if item.id == item_id), None)

    async def add_item(self, item: PlanItem) -> PlanItem:
        candidate = _clone_item(item)

        async def mutate(transaction: StateTransaction) -> PlanItem:
            current = await transaction.get_record(self._key)
            items = _decode_items(current)
            if any(existing.id == candidate.id for existing in items):
                raise ValueError(f"A step with id {candidate.id!r} is already in this plan.")
            items.append(candidate)
            await self._write(transaction, current, items)
            return _clone_item(candidate)

        return await self._store.mutate(mutate)

    async def update_item(
        self,
        item_id: str,
        *,
        content: "str | None" = None,
        status: "TaskStatus | None" = None,
        active_form: "str | None" = None,
        parent_id: "str | None" = None,
        depends_on: "list[str] | None" = None,
    ) -> "PlanItem | None":
        async def mutate(transaction: StateTransaction) -> "PlanItem | None":
            current = await transaction.get_record(self._key)
            items = _decode_items(current)
            index = next((i for i, item in enumerate(items) if item.id == item_id), None)
            if index is None:
                return None
            item = items[index]
            if content is not None:
                item.content = content
            if status is not None:
                item.status = status
            if active_form is not None:
                item.active_form = active_form
            if parent_id is not None:
                item.parent_id = parent_id
            if depends_on is not None:
                item.depends_on = list(depends_on)
            await self._write(transaction, current, items)
            return _clone_item(item)

        return await self._store.mutate(mutate)

    async def remove_item(self, item_id: str) -> bool:
        async def mutate(transaction: StateTransaction) -> bool:
            current = await transaction.get_record(self._key)
            items = _decode_items(current)
            next_items = [item for item in items if item.id != item_id]
            if len(next_items) == len(items):
                return False
            await self._write(transaction, current, next_items)
            return True

        return await self._store.mutate(mutate)

    async def _write(
        self,
        transaction: StateTransaction,
        current: "StoredRecord | None",
        items: list[PlanItem],
    ) -> None:
        next_record = self._record(_validated_items(items), current)
        if current is None:
            await transaction.insert_record(next_record)
            return
        if not await transaction.replace_record(
            next_record,
            expected_storage_version=current.storage_version,
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def _record(
        self,
        items: list[PlanItem],
        current: "StoredRecord | None",
    ) -> StoredRecord:
        data: dict[str, JsonValue] = {
            "version": 1,
            "owner_kind": self._owner_kind,
            "owner_id": self._owner_id,
            "items": [cast(JsonValue, item.model_dump(mode="json")) for item in items],
        }
        return StoredRecord(
            key_digest=self._key,
            partition_digest=self._partition,
            scope_digest=None,
            parent_digest=None,
            kind=_KIND,
            sort_key="plan:" + self._key.hex()[:64],
            state=None,
            storage_version=0 if current is None else current.storage_version,
            lease_owner=None,
            lease_fence=0,
            lease_expires_at=None,
            data=data,
        )


def _decode_items(record: "StoredRecord | None") -> list[PlanItem]:
    if record is None:
        return []
    data = record.data
    if data.get("version") != 1:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    items = data.get("items")
    if not isinstance(items, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        values = [PlanItem.model_validate(item) for item in items]
    except Exception as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    return _validated_items(values)


def _validated_items(items: list[PlanItem]) -> list[PlanItem]:
    values = [_clone_item(item) for item in items]
    ids = tuple(item.id for item in values)
    if len(ids) != len(set(ids)):
        raise ValueError("plan item ids must be unique")
    return values


def _clone_item(item: PlanItem) -> PlanItem:
    if not isinstance(item, PlanItem):
        raise TypeError("plan item must be PlanItem")
    return item.model_copy(deep=True)


__all__ = ["RuntimePlanStore"]
