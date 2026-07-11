#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lifecycle event dispatch.

``notify_remove`` only reconciles the running-container set on a *full*
container run; a partial removal leaves it untouched. This is a known
limitation, not fixed here to keep this module a pure behavior-preserving
move.
"""
import contextlib
import inspect
from typing import TYPE_CHECKING

from linktools.types import MISSING
from .hooks import HookPhase

if TYPE_CHECKING:
    from typing import Any
    from ..context import EventContext
    from ..manager import ContainerManager


class LifecycleDispatcher:
    """Dispatch on_check/on_starting/... lifecycle hooks behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def _invoke_callback(self, func, context: "Any" = MISSING) -> "Any":
        """Call an on_check/on_starting/... method: zero-arg if its signature
        takes no parameters (besides an already-bound self), otherwise with
        ``context``. Formerly ``ContainerManager._callback``, relocated here
        (Spec section 62) since dispatch is this class's sole concern."""
        if self.manager.environ.debug:
            self.manager.logger.debug(f"Callback {func}")
        if context is MISSING:
            return func()
        sig = inspect.signature(func)
        if len(sig.parameters) == 0:
            return func()
        else:
            return func(context)

    @contextlib.contextmanager
    def notify_start(self, context: "EventContext"):
        for container in context.target_containers:
            self._invoke_callback(container.on_check, context)
            container.hooks.call(HookPhase.CHECK, context)

        for container in context.target_containers:
            self._invoke_callback(container.on_starting, context)

        # Legacy start_hooks == container.hooks.legacy_view(BEFORE_START); a
        # hook registered directly through the registry (not via the legacy
        # `.append()` view) is picked up here too, in the same ordered bucket.
        for container in context.target_containers:
            container.hooks.call(HookPhase.BEFORE_START, context)

        self.manager.hooks.call(HookPhase.BEFORE_START, context)

        yield

        for container in reversed(context.target_containers):
            self._invoke_callback(container.on_started, context)
            container.hooks.call(HookPhase.AFTER_START, context, reverse=True)

    @contextlib.contextmanager
    def notify_stop(self, context: "EventContext"):
        for container in reversed(context.target_containers):
            self._invoke_callback(container.on_stopping, context)
            container.hooks.call(HookPhase.BEFORE_STOP, context, reverse=True)

        self.manager.hooks.call(HookPhase.BEFORE_STOP, context)

        yield

        for container in context.target_containers:
            self._invoke_callback(container.on_stopped, context)
            # Legacy stop_hooks == container.hooks.legacy_view(AFTER_STOP).
            container.hooks.call(HookPhase.AFTER_STOP, context)

        self.manager.hooks.call(HookPhase.AFTER_STOP, context)

    @contextlib.contextmanager
    def notify_remove(self, context: "EventContext"):
        yield

        if context.is_full_containers:
            running_names = self.manager.running_state.get_persisted()
            running_containers = [
                self.manager.containers[name] for name in running_names if name in self.manager.containers
            ]
            removed = [container for container in running_containers if container not in context.containers]
            for container in removed:
                # A removed container is no longer in the installed list, so its
                # `configs` defaults were never registered in env_config. Register
                # them here so on_removed can read its own configs without failing.
                self.manager.env_config.update_defaults(**container.configs)
                self._invoke_callback(container.on_removed, context)
                container.hooks.call(HookPhase.AFTER_REMOVE, context)
            self.manager.hooks.call(HookPhase.AFTER_REMOVE, context)
            self.manager.running_state.remove([container.name for container in removed])
