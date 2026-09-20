#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness adapters for Runtime-owned durable state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from linktools.core import environ
from pydantic_ai_harness.planning import (
    PlanItem as HarnessPlanItem,
)
from pydantic_ai_harness.planning import (
    TaskStatus,
)

from ..capability import ToolCallRetry
from ._plan import PlanItem, RuntimePlanStore

_logger = environ.get_logger("ai.runtime.harness")


class _InteractionStagingStore(Protocol):
    def intern_payload(self, run_id: str, payload: bytes) -> tuple[str, int]: ...

    def stage_model_interaction(self, interaction: object) -> None: ...


class HarnessPlanStoreAdapter:
    """Expose the Runtime plan store through Harness' public PlanStore contract."""

    def __init__(self, store: RuntimePlanStore) -> None:
        if not isinstance(store, RuntimePlanStore):
            raise TypeError("store must be RuntimePlanStore")
        self._store = store

    async def get_items(self) -> list[HarnessPlanItem]:
        return _harness_plan_items(await self._store.get_items())

    async def set_items(self, items: list[HarnessPlanItem]) -> None:
        await self._store.write_plan(_runtime_plan_items(items))

    async def get_item(self, item_id: str) -> HarnessPlanItem | None:
        return next(
            (item for item in await self.get_items() if item.id == item_id), None
        )

    async def add_item(self, item: HarnessPlanItem) -> HarnessPlanItem:
        candidate = item.model_copy(deep=True)

        def edit(current: list[PlanItem]) -> list[PlanItem]:
            values = _harness_plan_items(current)
            if any(value.id == candidate.id for value in values):
                raise ValueError(
                    f"A step with id {candidate.id!r} is already in this plan."
                )
            values.append(candidate.model_copy(deep=True))
            return _runtime_plan_items(values)

        await self._store.edit_items(edit)
        return candidate.model_copy(deep=True)

    async def update_item(
        self,
        item_id: str,
        *,
        content: str | None = None,
        status: TaskStatus | None = None,
        active_form: str | None = None,
        parent_id: str | None = None,
        depends_on: list[str] | None = None,
    ) -> HarnessPlanItem | None:
        def edit(current: list[PlanItem]) -> list[PlanItem]:
            values = _harness_plan_items(current)
            selected = next((item for item in values if item.id == item_id), None)
            if selected is None:
                return current
            if content is not None:
                selected.content = content
            if status is not None:
                selected.status = status
            if active_form is not None:
                selected.active_form = active_form
            if parent_id is not None:
                selected.parent_id = parent_id
            if depends_on is not None:
                selected.depends_on = list(depends_on)
            return _runtime_plan_items(values)

        previous, current, _revision = await self._store.edit_items(edit)
        if not any(item.id == item_id for item in _harness_plan_items(previous)):
            return None
        selected = next(
            (item for item in _harness_plan_items(current) if item.id == item_id),
            None,
        )
        if selected is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if active_form is not None:
            selected.active_form = active_form
        if depends_on is not None:
            selected.depends_on = list(depends_on)
        return selected.model_copy(deep=True)

    async def remove_item(self, item_id: str) -> bool:
        def edit(current: list[PlanItem]) -> list[PlanItem]:
            values = _harness_plan_items(current)
            return _runtime_plan_items(
                [item for item in values if item.id != item_id]
            )

        previous, _current, _revision = await self._store.edit_items(edit)
        return any(item.id == item_id for item in _harness_plan_items(previous))


def _harness_plan_items(items: Sequence[PlanItem]) -> list[HarnessPlanItem]:
    return [
        HarnessPlanItem(
            id=_plan_item_id(index),
            content=item.content,
            status=TaskStatus(item.status),
        )
        for index, item in enumerate(items)
    ]


def _runtime_plan_items(items: list[HarnessPlanItem]) -> list[PlanItem]:
    if not isinstance(items, list):
        raise ToolCallRetry(
            "Plan items are invalid for this runtime. Correct the plan and retry."
        )
    values: list[PlanItem] = []
    for item in items:
        if not isinstance(item, HarnessPlanItem):
            raise ToolCallRetry(
                "Plan items are invalid for this runtime. Correct the plan and retry."
            )
        if (
            not isinstance(item.parent_id, (str, type(None)))
            or not isinstance(item.depends_on, list)
            or any(not isinstance(value, str) for value in item.depends_on)
            or item.parent_id is not None
            or item.depends_on
        ):
            raise ToolCallRetry(
                "Subtasks and dependencies are not enabled. Use a flat plan and retry."
            )
        if not isinstance(item.status, TaskStatus):
            raise ToolCallRetry(
                "Plan items are invalid for this runtime. Correct the plan and retry."
            )
        if item.status is TaskStatus.blocked:
            raise ToolCallRetry(
                "Subtasks and dependencies are not enabled. Use a flat plan and retry."
            )
        if not isinstance(item.content, str) or not item.content.strip():
            raise ToolCallRetry(
                "Each plan item must have non-empty content. Fill every item and retry."
            )
        if item.status not in {
            TaskStatus.pending,
            TaskStatus.in_progress,
            TaskStatus.completed,
            TaskStatus.cancelled,
        }:
            raise ToolCallRetry(
                "Plan items are invalid for this runtime. Correct the plan and retry."
            )
        values.append(PlanItem(item.content, cast(str, item.status.value)))
    return values


def _plan_item_id(index: int) -> str:
    return f"item-{index + 1}"



__all__ = [
    "HarnessPlanStoreAdapter",
]
