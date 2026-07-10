#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lifecycle event dispatch (refactor spec Phase 4).

Extracted verbatim from ContainerManager.notify_start/notify_stop/notify_remove.
The hook/callback ordering is unchanged (refactor spec §8.5). ``notify_remove``'s
full-container-only cleanup logic is preserved as-is; its known limitation is not
fixed here to avoid mixing a behavior change with the code move.
"""
import contextlib
from typing import TYPE_CHECKING

from linktools.types import MISSING

if TYPE_CHECKING:
    from ..context import EventContext
    from ..manager import ContainerManager


class LifecycleDispatcher:
    """Dispatch on_check/on_starting/... lifecycle hooks behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    @contextlib.contextmanager
    def notify_start(self, context: "EventContext"):
        for container in context.target_containers:
            self.manager._callback(container.on_check, context)

        for container in context.target_containers:
            self.manager._callback(container.on_starting, context)

        for container in context.target_containers:
            if container.start_hooks:
                for hook in container.start_hooks:
                    self.manager._callback(hook)

        if self.manager.start_hooks:
            for hook in self.manager.start_hooks:
                self.manager._callback(hook)

        yield

        for container in reversed(context.target_containers):
            self.manager._callback(container.on_started, context)

    @contextlib.contextmanager
    def notify_stop(self, context: "EventContext"):
        for container in reversed(context.target_containers):
            self.manager._callback(container.on_stopping, context)

        yield

        for container in context.target_containers:
            self.manager._callback(container.on_stopped, context)
            if container.stop_hooks:
                for hook in container.stop_hooks:
                    self.manager._callback(hook)

        if self.manager.stop_hooks:
            for hook in self.manager.stop_hooks:
                self.manager._callback(hook)

    @contextlib.contextmanager
    def notify_remove(self, context: "EventContext"):
        yield

        if context.is_full_containers:
            with self.manager.environ.locks.process_lock("cntr:settings"):
                running_containers = self.manager._load_running_containers()
                all_containers = {*context.containers, *running_containers}
                for container in running_containers:
                    if container not in context.containers:
                        # A removed container is no longer in the installed list, so its
                        # `configs` defaults were never registered in env_config. Register
                        # them here so on_removed can read its own configs without failing.
                        self.manager.env_config.update_defaults(**container.configs)
                        self.manager._callback(container.on_removed, context)
                        all_containers.remove(container)
                self.manager._dump_running_containers(all_containers)
