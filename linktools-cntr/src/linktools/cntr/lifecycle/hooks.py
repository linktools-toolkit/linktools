#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed, ordered lifecycle hook registry.

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


class HookInvocation(str, Enum):
    """How a hook's callback is actually called -- fixed once at
    registration time (not re-derived on every call), so a callback whose
    signature can satisfy neither calling convention is rejected up front
    instead of raising a confusing TypeError deep inside a real
    up/restart/down."""
    NO_ARGS = "no-args"
    CONTEXT = "context"


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
    # reference listed here instead is allowed and only produces a warning.
    optional_before: "tuple" = ()
    optional_after: "tuple" = ()
    source: "str | None" = None
    opaque: bool = False
    invocation: HookInvocation = HookInvocation.NO_ARGS
    metadata: "dict[str, Any]" = field(default_factory=dict)


def _can_bind(signature: "inspect.Signature", *args) -> bool:
    try:
        signature.bind(*args)
    except TypeError:
        return False
    return True


_SENTINEL_CONTEXT = object()


def _resolve_invocation(callback: "Callable") -> "HookInvocation | None":
    """``None`` means the signature could not be introspected at all (some
    builtins) -- the caller falls back to opaque/NO_ARGS for those.

    Raises ``HookValidationError`` if the signature can satisfy neither a
    zero-arg nor a one-arg (context) call -- e.g. two required positional
    parameters, or a required keyword-only parameter."""
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return None
    can_context = _can_bind(signature, _SENTINEL_CONTEXT)
    can_no_args = _can_bind(signature)
    if can_context:
        # Preferred: a hook that can take an optional context receives one.
        return HookInvocation.CONTEXT
    if can_no_args:
        return HookInvocation.NO_ARGS
    raise HookValidationError(
        f"Hook callback {callback!r} must accept either zero arguments or exactly "
        f"one (context); its signature {signature} accepts neither"
    )


def _is_json_compatible(value) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_compatible(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_compatible(v) for k, v in value.items())
    return False


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
        if key is not None:
            try:
                hash(key)
            except TypeError as exc:
                raise HookValidationError(f"Hook key must be hashable: {key!r}") from exc
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

        metadata = dict(metadata or {})
        if not _is_json_compatible(metadata):
            raise HookValidationError(f"Hook metadata must be JSON-compatible: {metadata!r}")

        invocation = HookInvocation.NO_ARGS
        if not opaque:
            resolved = _resolve_invocation(callback)
            if resolved is None:
                # Signature cannot be introspected (e.g. some builtins) --
                # fall back to the always-zero-arg compatible calling strategy.
                opaque = True
            else:
                invocation = resolved

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
            invocation=invocation,
            metadata=metadata,
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
        returned list."""
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

        Always validates ``phase`` first: a missing *required* before/after
        reference or an ordering cycle must fail before any hook runs, not
        be silently skipped.

        A CONTEXT-invocation hook receives ``context`` (when one is given);
        a NO_ARGS (or opaque/legacy) hook is always called with no
        arguments, so every hook reachable through the legacy
        start_hooks/stop_hooks view is invoked zero-arg.
        """
        self.validate(phase)
        hooks = list(self.iter_phase(phase))
        if reverse:
            hooks = list(reversed(hooks))
        for hook in hooks:
            self._invoke(hook, context)

    def _invoke(self, hook: "Hook", context: "Any") -> "Any":
        if hook.invocation == HookInvocation.CONTEXT and context is not None:
            return hook.callback(context)
        return hook.callback()


class HookListView(MutableSequence):
    """``MutableSequence`` facade over one ``(registry, phase)`` bucket.

    Preserves every existing ``start_hooks``/``stop_hooks`` usage
    (``append``/``extend``/``insert``/``remove``/``pop``/``clear``/``len``/
    ``bool``/indexing/iteration); reads and writes the raw callback, never a
    ``Hook`` object.

    Registry ordering is by (topology, order, registration position), not
    raw list index -- a formal hook's dependency-driven position can never
    be overridden by a legacy `insert()`. Every hook this view itself
    creates (``append``/``insert``/index-assignment) shares one
    ``source="legacy"`` segment, and only that segment's members are ever
    repositioned to honor an explicit ``insert(index, ...)``.
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
        if isinstance(index, slice):
            start, _stop, step = index.indices(len(hooks))
            targets = hooks[index]
            if step != 1:
                values = list(value)
                if len(values) != len(targets):
                    raise ValueError(
                        f"attempt to assign sequence of size {len(values)} "
                        f"to extended slice of size {len(targets)}"
                    )
                for hook, v in zip(targets, values):
                    self._registry.unregister(self._phase, hook.key)
                    self._registry.register(
                        self._phase, v, key=hook.key, name=hook.name, order=hook.order,
                        source=hook.source, opaque=True,
                    )
                return
            # Contiguous slice: remove the old range, then insert the new
            # values at its start position, preserving relative order.
            del self[index]
            for offset, v in enumerate(value):
                self.insert(start + offset, v)
            return

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
        """Insert ``value`` so that, among this view's own (legacy) hooks,
        it ends up exactly at ``index`` -- existing legacy hooks before
        ``index`` in the current view stay before it, the rest stay after.
        Formal (keyed) hooks are never reordered; their dict position (and
        so their same-``order`` tie-break) is left untouched."""
        bucket = self._registry._hooks.setdefault(self._phase, {})
        current = self._hooks()
        index = max(0, min(index, len(current)))
        before_keys = [hook.key for hook in current[:index] if hook.source == "legacy"]
        after_keys = [hook.key for hook in current[index:] if hook.source == "legacy"]

        def _requeue(key):
            bucket[key] = bucket.pop(key)

        for key in before_keys:
            _requeue(key)
        self._registry.register(self._phase, value, source="legacy", opaque=True)
        for key in after_keys:
            _requeue(key)

    def append(self, value):
        self._registry.register(self._phase, value, source="legacy", opaque=True)
