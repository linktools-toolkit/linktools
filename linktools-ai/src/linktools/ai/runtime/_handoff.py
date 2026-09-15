#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory coordination for transient runtime handoff."""

import asyncio
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

KeyT = TypeVar("KeyT", bound=Hashable)
ValueT = TypeVar("ValueT")


@dataclass(slots=True)
class HandoffState(Generic[ValueT]):
    active_consumers: int = 0
    dependency_holds: set[str] = field(default_factory=set)
    release_requested: bool = False
    release_in_progress: bool = False
    release_value: "ValueT | None" = None


class HandoffGate(Generic[KeyT, ValueT]):
    """Coordinate consumers and one transient cleanup owner per resource."""

    def __init__(self) -> None:
        self._states: dict[KeyT, HandoffState[ValueT]] = {}
        self._condition = asyncio.Condition()

    async def enter(self, key: KeyT) -> HandoffState[ValueT]:
        async with self._condition:
            while True:
                state = self._states.get(key)
                if state is None:
                    state = HandoffState()
                    self._states[key] = state
                if not state.release_in_progress:
                    state.active_consumers += 1
                    return state
                await self._condition.wait()

    async def leave(self, key: KeyT, state: HandoffState[ValueT]) -> bool:
        async with self._condition:
            if self._states.get(key) is not state:
                raise RuntimeError("handoff state changed while consumer was active")
            state.active_consumers -= 1
            if state.active_consumers < 0:
                raise RuntimeError("handoff consumer count became negative")
            owner = self._claim_cleanup(state)
            self._discard_idle(key, state, owner=owner)
            self._condition.notify_all()
            return owner

    async def acquire_hold(self, key: KeyT, hold_id: str) -> None:
        if not isinstance(hold_id, str) or not hold_id.strip():
            raise ValueError("handoff dependency hold id is required")
        async with self._condition:
            while True:
                state = self._states.get(key)
                if state is None:
                    state = HandoffState()
                    self._states[key] = state
                if not state.release_in_progress:
                    state.dependency_holds.add(hold_id)
                    return
                await self._condition.wait()

    async def release_hold(
        self, key: KeyT, hold_id: str
    ) -> "tuple[HandoffState[ValueT] | None, bool]":
        async with self._condition:
            state = self._states.get(key)
            if state is None:
                return None, False
            state.dependency_holds.discard(hold_id)
            owner = self._claim_cleanup(state)
            self._discard_idle(key, state, owner=owner)
            self._condition.notify_all()
            return state, owner

    async def request_release(
        self,
        key: KeyT,
        *,
        value: "ValueT | None" = None,
        require_existing: bool = False,
    ) -> tuple[HandoffState[ValueT], bool]:
        async with self._condition:
            state = self._states.get(key)
            if state is None:
                if require_existing:
                    raise RuntimeError("handoff release requested without consumer")
                state = HandoffState()
                self._states[key] = state
            state.release_requested = True
            state.release_value = value
            owner = self._claim_cleanup(state)
            self._condition.notify_all()
            return state, owner

    async def finish_release(
        self,
        key: KeyT,
        state: HandoffState[ValueT],
        *,
        succeeded: bool,
    ) -> None:
        async with self._condition:
            if self._states.get(key) is not state:
                self._condition.notify_all()
                return
            if succeeded and state.active_consumers == 0 and not state.dependency_holds:
                self._states.pop(key, None)
            else:
                state.release_in_progress = False
                state.release_requested = True
            self._condition.notify_all()

    @staticmethod
    def _claim_cleanup(state: HandoffState[ValueT]) -> bool:
        if (
            state.active_consumers == 0
            and not state.dependency_holds
            and state.release_requested
            and not state.release_in_progress
        ):
            state.release_in_progress = True
            return True
        return False

    def _discard_idle(
        self,
        key: KeyT,
        state: HandoffState[ValueT],
        *,
        owner: bool,
    ) -> None:
        if (
            not owner
            and not state.release_requested
            and not state.dependency_holds
            and state.active_consumers == 0
            and self._states.get(key) is state
        ):
            self._states.pop(key, None)


__all__: list[str] = []
