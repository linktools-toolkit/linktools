#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optional owner-local atomic persistence contracts for capabilities."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from ...core import CommitPolicy, JsonValue
from ...storage import StorageRoute
from ._store import StateTransaction


class CapabilityWriter(Protocol):
    """Borrow one run's writer; deferred acceptance is not a durable receipt."""

    async def submit(self, *, key: str, payload: Mapping[str, JsonValue]) -> None: ...

    async def flush(self) -> None:
        """Confirm accepted compatible writes or raise, retaining unresolved work."""
        ...


@runtime_checkable
class CapabilityPersistence(Protocol):
    """Opt into same-run atomic writes with retained failures and explicit retry.

    Transaction callbacks must have no external side effects. Bound writers and
    pending callbacks are owner-local, not portable distributed queue consumers.
    """

    @property
    def storage(self) -> StorageRoute | None: ...

    @property
    def commit(self) -> CommitPolicy: ...

    def bind_persistence(self, writer: CapabilityWriter) -> "CapabilityPersistence":
        """Return a separate run-local capability instance using this writer."""
        ...

    async def persist_in_transaction(
        self, transaction: StateTransaction, payload: Mapping[str, JsonValue],
    ) -> None: ...


__all__ = ["CapabilityPersistence", "CapabilityWriter"]
