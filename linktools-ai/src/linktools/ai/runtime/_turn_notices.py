#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Host-owned advisory delivery for committed ordinary turn milestones."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Literal, Protocol

from ..core import validate_persistence_namespace, validate_resource_id, validate_tenant_id


@dataclass(frozen=True, slots=True)
class TurnNoticeScope:
    """Exact identity of a Runtime's ordinary conversation space."""

    namespace: str
    tenant_id: str

    def __post_init__(self) -> None:
        validate_persistence_namespace(self.namespace)
        validate_tenant_id(self.tenant_id)


@dataclass(frozen=True, slots=True)
class TurnNotice:
    """An advisory hint only; no prompt, result, lease or actor information."""

    scope: TurnNoticeScope
    session_id: str
    turn_id: str
    revision: int
    phase: Literal["admitted", "terminal", "cancelling"]

    def __post_init__(self) -> None:
        validate_resource_id(self.session_id)
        validate_resource_id(self.turn_id)
        if type(self.revision) is not int or self.revision < 1 or self.phase not in ("admitted", "terminal", "cancelling"):
            raise ValueError("invalid turn notice")


class TurnNoticeTransport(Protocol):
    """Borrowed multi-producer, fan-out channel for one Runtime scope.

    The host owns transport credentials and backpressure. Offer must not block:
    use immediate delivery or a bounded queue. Callbacks run on the subscribing
    event loop. On reconnect, known loss or overflow the carrier must call
    `on_gap(scope)` for affected subscriptions, even when their local browser
    streams remain open. A gap asks those clients to re-read authorized state.
    Subscribing must not expose a caller-controlled scope or confer read access.
    """

    def subscribe(
        self, scope: TurnNoticeScope, *,
        on_notice: Callable[[TurnNotice], None],
        on_gap: Callable[[TurnNoticeScope], None],
    ) -> AbstractAsyncContextManager[None]:
        """Enter when subscribed; exit detaches all callbacks and pending reads."""
        ...

    def offer(self, notice: TurnNotice) -> None:
        """Offer a post-commit notice without waiting for remote acknowledgement."""
        ...


__all__ = ["TurnNotice", "TurnNoticeScope", "TurnNoticeTransport"]
