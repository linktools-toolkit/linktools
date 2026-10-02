#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lifecycle event dispatch."""
import contextlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

from linktools.types import MISSING
from ..container import ContainerError
from .hooks import HookPhase

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import Any
    from ..container import BaseContainer
    from ..context import EventContext
    from ..manager import ContainerManager


@dataclass(frozen=True)
class LifecycleStep:
    """One callback and/or registry bucket; ``container=None`` means manager.

    Callback names are resolved only during dispatch, so describing a
    lifecycle never invokes container code or snapshots its mutable hooks.
    """

    container: "BaseContainer | None" = None
    phase: "HookPhase | None" = None
    callback: "str | None" = None
    reverse: bool = False


class LifecycleDispatcher:
    """Dispatch on_check/on_starting/... lifecycle hooks behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def _invoke_callback(self, func, context: "Any" = MISSING) -> "Any":
        """Call an on_check/on_starting/... method: zero-arg if its signature
        takes no parameters (besides an already-bound self), otherwise with
        ``context``."""
        if self.manager.environ.debug:
            self.manager.logger.debug(f"Callback {func}")
        if context is MISSING:
            return func()
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            return func()
        else:
            return func(context)

    @classmethod
    def iter_steps(
            cls,
            action: str,
            containers: "Sequence[BaseContainer]",
            after: "bool | None" = None,
    ) -> "Iterator[LifecycleStep]":
        """Describe dispatch order without preparing containers or calling hooks.

        ``after`` selects one side of the runtime operation; the default
        describes both. Registry contents stay live until each step is
        consumed, including hooks registered by earlier callbacks.
        """
        if action == "restart":
            yield from cls.iter_steps("down", containers, after=after)
            yield from cls.iter_steps("up", containers, after=after)
        elif action == "up":
            if after is not True:
                for container in containers:
                    yield LifecycleStep(container, HookPhase.CHECK, "on_check")
                # Every on_starting may register hooks on another container.
                # Finish all callbacks before looking up BEFORE_START buckets.
                for container in containers:
                    yield LifecycleStep(container, callback="on_starting")
                for container in containers:
                    yield LifecycleStep(container, HookPhase.BEFORE_START)
                yield LifecycleStep(phase=HookPhase.BEFORE_START)
            if after is not False:
                for container in reversed(containers):
                    yield LifecycleStep(container, HookPhase.AFTER_START, "on_started", reverse=True)
        elif action == "down":
            if after is not True:
                for container in reversed(containers):
                    yield LifecycleStep(container, HookPhase.BEFORE_STOP, "on_stopping", reverse=True)
                yield LifecycleStep(phase=HookPhase.BEFORE_STOP)
            if after is not False:
                for container in containers:
                    yield LifecycleStep(container, HookPhase.AFTER_STOP, "on_stopped")
                yield LifecycleStep(phase=HookPhase.AFTER_STOP)
        else:
            raise ContainerError(f"Unsupported lifecycle action: {action!r}; expected up/restart/down")

    def _dispatch_steps(self, steps: "Iterable[LifecycleStep]", context: "EventContext") -> None:
        for step in steps:
            if step.callback is not None:
                self._invoke_callback(getattr(step.container, step.callback), context)
            if step.phase is not None:
                owner = step.container if step.container is not None else self.manager
                owner.hooks.call(step.phase, context, reverse=step.reverse)

    @contextlib.contextmanager
    def notify_start(self, context: "EventContext") -> "Iterator[None]":
        self._dispatch_steps(self.iter_steps("up", context.target_containers, after=False), context)
        yield
        self._dispatch_steps(self.iter_steps("up", context.target_containers, after=True), context)

    @contextlib.contextmanager
    def notify_stop(self, context: "EventContext") -> "Iterator[None]":
        self._dispatch_steps(self.iter_steps("down", context.target_containers, after=False), context)
        yield
        self._dispatch_steps(self.iter_steps("down", context.target_containers, after=True), context)

    @contextlib.contextmanager
    def notify_remove(self, context: "EventContext") -> "Iterator[None]":
        yield

        # context.containers is always the FULL installed project (see
        # ComposeOperations._make_context: it's built from
        # selection.project_containers, never narrowed to the partial
        # target set) -- so comparing it against the persisted running set
        # is safe after every lifecycle operation, not just a full one. A
        # partial up/down/restart must also reconcile a container that was
        # removed from the installed set since it was last marked running.
        running_names = self.manager.running_state.get_persisted()
        running_containers = [
            self.manager.containers[name] for name in running_names if name in self.manager.containers
        ]
        removed = [container for container in running_containers if container not in context.containers]
        if not removed:
            return
        for container in removed:
            # A removed container is no longer in the installed list, so its
            # `configs` defaults were never registered -- register them now
            # so on_removed can read its own configs without failing.
            container.register_configs()
            self._invoke_callback(container.on_removed, context)
            container.hooks.call(HookPhase.AFTER_REMOVE, context)
        self.manager.hooks.call(HookPhase.AFTER_REMOVE, context)
        self.manager.running_state.remove([container.name for container in removed])
