#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Running-container state, updated by ComposeRunner/CLI/exec *after* a
successful compose up/down:

- partial up  -> add target names
- partial down -> remove target names
- full up     -> set persisted = target names (the actual target set)
- full down   -> clear persisted

``get_actual`` (live ``docker compose ps``) is not implementable today because the
runtime ``popen`` wrapper has no output capture, so it raises
``RuntimeStateUnavailable``. ``get_effective`` falls back to the persisted state,
so ``ct-cntr list`` never crashes when Docker is absent. When output capture is
added, only ``get_actual`` needs to change.
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from ..container import BaseContainer
    from ..context import EventContext
    from ..manager import ContainerManager


_RUNNING_KEY = "RUNNING_CONTAINERS"


class RuntimeStateUnavailable(Exception):
    """Raised when live container runtime state cannot be queried."""


class RunningStateStore:
    """Owns the persisted set of running container names behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def _get(self) -> "list[str]":
        # A failed migration is not cached, so retry it on every access.
        self.manager._migrated
        return list(self.manager._transient_ns.get(_RUNNING_KEY, []) or [])

    def _set(self, names: "Iterable[str]") -> None:
        self.manager._migrated
        self.manager._transient_ns.set(_RUNNING_KEY, sorted(set(names)))

    def get_persisted(self) -> "list[str]":
        """Names recorded as running in the persisted store."""
        return self._get()

    def get_actual(self, containers: "Iterable[BaseContainer]") -> "list[str]":
        """Live running names via ``docker compose ps``.

        Not implemented: the runtime popen wrapper has no output capture.
        Raises so callers fall back to persisted state.
        """
        raise RuntimeStateUnavailable("docker compose ps output capture is not supported")

    def get_effective(self, containers: "Iterable[BaseContainer]") -> "list[str]":
        """Prefer live state; fall back to persisted when it is unavailable."""
        try:
            return self.get_actual(containers)
        except RuntimeStateUnavailable:
            return self.get_persisted()

    def mark_started(self, context: "EventContext") -> None:
        """Record the context's target containers as running (after a successful up)."""
        targets = [c.name for c in context.target_containers]
        if context.is_full_containers:
            # Full up writes the actual target set (drops anything no longer installed).
            self._set(targets)
        else:
            self._set(set(self._get()) | set(targets))

    def mark_stopped(self, context: "EventContext") -> None:
        """Record the context's target containers as stopped (after a successful down)."""
        targets = {c.name for c in context.target_containers}
        if context.is_full_containers:
            # Full down stops everything -> clear the persisted running set.
            self._set([])
        else:
            self._set(set(self._get()) - targets)
