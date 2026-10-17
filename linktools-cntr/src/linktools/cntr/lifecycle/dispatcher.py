#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Existing lifecycle callbacks with explicit preparation and check boundaries."""
import contextlib
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING

from linktools.types import MISSING
from ..errors import ContainerError
from .hooks import HookPhase

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence
    from typing import Any
    from ..container import BaseContainer
    from ..context import OperationContext
    from ..manager import ContainerManager


@dataclass(frozen=True)
class LifecycleStep:
    container: "BaseContainer | None" = None
    phase: "HookPhase | None" = None
    callback: "str | None" = None
    reverse: bool = False


class LifecycleDispatcher:
    def __init__(self, manager: "ContainerManager") -> None:
        self.manager = manager

    def _invoke_callback(self, callback, context: "Any" = MISSING) -> "Any":
        if context is MISSING or len(inspect.signature(callback).parameters) == 0:
            return callback()
        return callback(context)

    def prepare(self, containers: "Iterable[BaseContainer]") -> None:
        """Prepare installed declarations before constructing an execution model."""
        for container in containers:
            container.reset_rendered_inputs()
            self._invoke_callback(container.on_prepare)

    @classmethod
    def _preparation_steps(cls, containers):
        for container in containers:
            yield LifecycleStep(container, callback="on_starting")
        for container in containers:
            yield LifecycleStep(container, HookPhase.BEFORE_START)
        yield LifecycleStep(phase=HookPhase.BEFORE_START)

    def prepare_start(self, context: "OperationContext") -> None:
        """Initialize only explicitly selected owners before comparing consumers."""
        self._dispatch_steps((LifecycleStep(container, callback="on_starting")
                              for container in tuple(context.target_containers)), context)

    @classmethod
    def iter_steps(cls, action: str, containers: "Sequence[BaseContainer]",
                   after: "bool | None" = None,
                   stop_containers: "Sequence[BaseContainer] | None" = None) -> "Iterator[LifecycleStep]":
        containers = tuple(containers)
        if action == "restart":
            if after is not True:
                yield from cls.iter_steps("up", containers, after=False)
                yield from cls.iter_steps("down", containers if stop_containers is None else stop_containers)
            if after is not False:
                yield from cls.iter_steps("up", containers, after=True)
        elif action == "up":
            if after is not True:
                yield from cls._preparation_steps(containers)
                for container in containers:
                    yield LifecycleStep(container, HookPhase.CHECK, "on_check")
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
            raise ContainerError("Unsupported lifecycle action: " + action)

    def _dispatch_steps(self, steps: "Iterable[LifecycleStep]", context: "OperationContext") -> None:
        for step in steps:
            if step.callback is not None:
                self._invoke_callback(getattr(step.container, step.callback), context)
            if step.phase is not None:
                owner = step.container if step.container is not None else self.manager
                owner.hooks.call(step.phase, context, reverse=step.reverse)

    def check(self, context: "OperationContext") -> None:
        """Check prepared inputs against ready images before service mutation."""
        self._dispatch_steps((LifecycleStep(container, HookPhase.CHECK, "on_check")
                              for container in tuple(context.target_containers)), context)

    @contextlib.contextmanager
    def notify_start(self, context: "OperationContext", *, prepare: bool = True,
                     generated_only: "Iterable[BaseContainer]" = ()) -> "Iterator[None]":
        """Notify application targets; skip preparation after a prior prepare_start."""
        targets = tuple(context.target_containers)
        if prepare:
            self.prepare_start(context)
        generated_only = set(generated_only)
        for container in targets:
            if container in generated_only:
                from ..artifacts import prepared_hooks
                from pathlib import Path
                prepared = prepared_hooks(container)
                files = {str(Path(hook.metadata["path"]) / name)
                         for hook in prepared for name in hook.metadata.get("files", ())}
                for hook in container.hooks.iter_phase(HookPhase.BEFORE_START):
                    key = hook.key
                    ownership = (hook.source == "builtin" and isinstance(key, tuple) and
                                 len(key) == 5 and key[0] == "chown" and
                                 key[1] in files and key[3] is False)
                    if hook in prepared or ownership:
                        self._invoke_callback(hook.callback, context)
            else:
                container.hooks.call(HookPhase.BEFORE_START, context)
        self.manager.hooks.call(HookPhase.BEFORE_START, context)
        yield
        try:
            self._dispatch_steps(self.iter_steps("up", targets, after=True), context)
        except Exception as error:
            raise ContainerError("Services were applied; after-start callback failed: {}".format(error)) from error

    @contextlib.contextmanager
    def notify_stop(self, context: "OperationContext") -> "Iterator[None]":
        targets = tuple(context.target_containers)
        self._dispatch_steps(self.iter_steps("down", targets, after=False), context)
        yield
        self._dispatch_steps(self.iter_steps("down", targets, after=True), context)

    def reconcile_removed(self, context: "OperationContext") -> None:
        running_names = self.manager.running_state.get_persisted()
        removed = [self.manager.containers[name] for name in running_names
                   if name in self.manager.containers and
                   self.manager.containers[name] not in context.project_containers]
        if not removed:
            return
        for container in removed:
            container.register_fields()
            self._invoke_callback(container.on_removed, context)
            container.hooks.call(HookPhase.AFTER_REMOVE, context)
        self.manager.hooks.call(HookPhase.AFTER_REMOVE, context)
        self.manager.running_state.remove([container.name for container in removed])
