#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed, ordered lifecycle hook registry (Spec Part X).

Replaces the bare ``list[Callable]`` that used to back
``container.start_hooks``/``stop_hooks``/``manager.start_hooks``/``stop_hooks``
with a registry that can validate identity, phase, ordering and before/after
constraints, while keeping every existing ``.append()``/iteration/indexing
usage working unchanged through ``HookListView``.
"""
import inspect
import itertools
from collections.abc import MutableSequence
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ..container import ContainerError

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Sequence
    from typing import Any


class HookPhase(str, Enum):
    CHECK = "check"
    BEFORE_START = "before-start"
    AFTER_START = "after-start"
    BEFORE_STOP = "before-stop"
    AFTER_STOP = "after-stop"
    AFTER_REMOVE = "after-remove"


class HookError(ContainerError):
    pass


class HookValidationError(HookError):
    pass


class HookOrderError(HookValidationError):
    pass


class HookCycleError(HookOrderError):
    pass


@dataclass(frozen=True)
class Hook:
    phase: HookPhase
    key: "Hashable"
    callback: "Callable"
    name: str
    order: int = 500
    before: "tuple" = ()
    after: "tuple" = ()
    # A missing before/after reference is a validation error by default; a
    # reference listed here instead is allowed and only produces a warning
    # (Spec section 58: "如果目标明确标记 optional").
    optional_before: "tuple" = ()
    optional_after: "tuple" = ()
    source: "str | None" = None
    opaque: bool = False
    metadata: "dict[str, Any]" = field(default_factory=dict)


def _accepts_context(callback) -> bool:
    """True if ``callback``'s signature declares >=1 positional parameter.

    A callback whose signature cannot be introspected at all (e.g. some
    builtins) is treated as opaque/zero-arg by the caller, not here.
    """
    sig = inspect.signature(callback)
    params = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(params) >= 1


class HookRegistry:
    """Owns one owner's (a container or the manager) hooks across all phases."""

    def __init__(self, owner: "Any" = None, scope: "str | None" = None):
        self.owner = owner
        self.scope = scope
        self._hooks: "dict[HookPhase, dict[Hashable, Hook]]" = {}
        self._sequence = itertools.count()

    def register(
            self,
            phase,
            callback: "Callable",
            key: "Hashable | None" = None,
            name: "str | None" = None,
            order: int = 500,
            before: "Sequence[Hashable]" = (),
            after: "Sequence[Hashable]" = (),
            optional_before: "Sequence[Hashable]" = (),
            optional_after: "Sequence[Hashable]" = (),
            source: "str | None" = None,
            opaque: bool = False,
            metadata: "dict[str, Any] | None" = None,
    ) -> "Hook":
        if not callable(callback):
            raise HookValidationError(f"Hook callback must be callable: {callback!r}")
        phase = HookPhase(phase)
        bucket = self._hooks.setdefault(phase, {})

        if key is None:
            # Legacy/anonymous append: a local, non-persistable identity that
            # never collides and is never deduplicated against.
            key = ("__sequence__", next(self._sequence))
        elif key in bucket:
            # Idempotent: the same phase+key registers once (template
            # re-render dedup); first registration wins.
            return bucket[key]

        before = tuple(before or ())
        after = tuple(after or ())
        optional_before = tuple(optional_before or ())
        optional_after = tuple(optional_after or ())
        if key in (*before, *after, *optional_before, *optional_after):
            raise HookCycleError(f"Hook {key!r} cannot reference itself in before/after")

        if not opaque:
            try:
                _accepts_context(callback)
            except (TypeError, ValueError):
                # Signature cannot be introspected (e.g. some builtins) --
                # fall back to the always-zero-arg compatible calling strategy.
                opaque = True

        hook = Hook(
            phase=phase,
            key=key,
            callback=callback,
            name=name or getattr(callback, "__name__", repr(callback)),
            order=order,
            before=before,
            after=after,
            optional_before=optional_before,
            optional_after=optional_after,
            source=source,
            opaque=opaque,
            metadata=dict(metadata or {}),
        )
        bucket[key] = hook
        return hook

    def unregister(self, phase, key: "Hashable") -> None:
        phase = HookPhase(phase)
        self._hooks.get(phase, {}).pop(key, None)

    def get(self, phase, key: "Hashable") -> "Hook | None":
        phase = HookPhase(phase)
        return self._hooks.get(phase, {}).get(key)

    def iter_phase(self, phase) -> "Iterator[Hook]":
        yield from self._ordered(HookPhase(phase))

    def _ordered(self, phase: "HookPhase") -> "list[Hook]":
        bucket = self._hooks.get(phase, {})
        if not bucket:
            return []
        keys = list(bucket.keys())
        registration_order = {key: index for index, key in enumerate(keys)}

        # before/after are folded into one "must come after" dependency graph:
        # `after=(x,)` means x must run first; `before=(x,)` means x must run
        # after this hook, i.e. this hook is a dependency of x.
        depends_on: "dict[Hashable, set]" = {key: set() for key in keys}
        for hook in bucket.values():
            for other in (*hook.after, *hook.optional_after):
                if other in bucket:
                    depends_on[hook.key].add(other)
            for other in (*hook.before, *hook.optional_before):
                if other in bucket:
                    depends_on[other].add(hook.key)

        def sort_key(k):
            return (bucket[k].order, registration_order[k])

        result: "list[Hook]" = []
        visited: "set" = set()
        visiting: "set" = set()

        def visit(key):
            if key in visited:
                return
            if key in visiting:
                raise HookCycleError(f"Cycle detected in hook ordering for phase {phase}: {key!r}")
            visiting.add(key)
            for dep in sorted(depends_on[key], key=sort_key):
                visit(dep)
            visiting.discard(key)
            visited.add(key)
            result.append(bucket[key])

        for key in sorted(keys, key=sort_key):
            visit(key)
        return result

    def validate(self, phase=None) -> "list[str]":
        """Raise on a missing *required* before/after reference or an
        ordering cycle; a missing reference listed in optional_before/
        optional_after instead only contributes a warning string to the
        returned list (Spec section 58)."""
        warnings: "list[str]" = []
        phases = [HookPhase(phase)] if phase is not None else list(self._hooks.keys())
        for ph in phases:
            bucket = self._hooks.get(ph, {})
            keys = set(bucket.keys())
            for hook in bucket.values():
                for other in (*hook.before, *hook.after):
                    if other not in keys:
                        raise HookValidationError(
                            f"Hook {hook.key!r} references unknown hook {other!r} in phase {ph}"
                        )
                for other in (*hook.optional_before, *hook.optional_after):
                    if other not in keys:
                        warnings.append(
                            f"Hook {hook.key!r} references unknown optional hook {other!r} in phase {ph}"
                        )
            self._ordered(ph)  # raises HookCycleError on a cycle
        return warnings

    def describe(self, phase=None) -> "list[dict[str, Any]]":
        phases = [HookPhase(phase)] if phase is not None else list(HookPhase)
        result = []
        for ph in phases:
            for hook in self._ordered(ph):
                result.append(dict(
                    phase=hook.phase.value,
                    key=repr(hook.key),
                    name=hook.name,
                    order=hook.order,
                    before=[repr(k) for k in hook.before],
                    after=[repr(k) for k in hook.after],
                    scope=self.scope,
                    source=hook.source,
                    opaque=hook.opaque,
                    metadata=dict(hook.metadata),
                ))
        return result

    def legacy_view(self, phase) -> "HookListView":
        return HookListView(self, HookPhase(phase))

    def call(self, phase, context: "Any" = None, reverse: bool = False) -> None:
        """Invoke every hook registered for ``phase``, in registry order.

        A hook accepting >=1 positional parameter receives ``context``; a
        0-arg (or opaque/legacy) hook is called with no arguments -- matching
        the pre-registry ``manager._callback`` calling convention exactly, so
        every hook reachable through the legacy start_hooks/stop_hooks view
        keeps being invoked zero-arg.
        """
        hooks = list(self.iter_phase(phase))
        if reverse:
            hooks = list(reversed(hooks))
        for hook in hooks:
            self._invoke(hook, context)

    def _invoke(self, hook: "Hook", context: "Any") -> "Any":
        callback = hook.callback
        if hook.opaque or context is None or not _accepts_context(callback):
            return callback()
        return callback(context)


class HookListView(MutableSequence):
    """``MutableSequence`` facade over one ``(registry, phase)`` bucket.

    Preserves every existing ``start_hooks``/``stop_hooks`` usage
    (``append``/``extend``/``insert``/``remove``/``pop``/``clear``/``len``/
    ``bool``/indexing/iteration); reads and writes the raw callback, never a
    ``Hook`` object.
    """

    def __init__(self, registry: "HookRegistry", phase: "HookPhase"):
        self._registry = registry
        self._phase = phase

    def _hooks(self) -> "list[Hook]":
        return self._registry._ordered(self._phase)

    def __len__(self):
        return len(self._hooks())

    def __getitem__(self, index):
        hooks = self._hooks()
        if isinstance(index, slice):
            return [hook.callback for hook in hooks[index]]
        return hooks[index].callback

    def __setitem__(self, index, value):
        hooks = self._hooks()
        hook = hooks[index]
        self._registry.unregister(self._phase, hook.key)
        self._registry.register(
            self._phase, value, key=hook.key, name=hook.name, order=hook.order,
            source=hook.source, opaque=True,
        )

    def __delitem__(self, index):
        hooks = self._hooks()
        targets = hooks[index] if isinstance(index, slice) else [hooks[index]]
        for hook in targets:
            self._registry.unregister(self._phase, hook.key)

    def __bool__(self):
        return len(self) > 0

    def __repr__(self):
        return repr([hook.callback for hook in self._hooks()])

    def insert(self, index, value):
        # Registry ordering is by (before/after, order, registration
        # sequence), not list position -- an explicit numeric `index` cannot
        # be honored exactly. Every existing caller only ever uses `insert`
        # via list-like `.append()`, so this registers `value` as a new
        # opaque, unkeyed legacy hook appended after the existing ones.
        self._registry.register(self._phase, value, source="legacy", opaque=True)

    def append(self, value):
        self._registry.register(self._phase, value, source="legacy", opaque=True)
