#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed lifecycle hook registration, validation, ordering and invocation."""
import inspect
import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from ..container import ContainerError

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator, Sequence
    from typing import Any


class HookPhase(str, Enum):
    AFTER_COMPOSE_RENDER = "after-compose-render"
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
            phase: "HookPhase | str",
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
            # Anonymous registrations have distinct local identities.
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

    def unregister(self, phase: "HookPhase | str", key: "Hashable") -> None:
        phase = HookPhase(phase)
        self._hooks.get(phase, {}).pop(key, None)

    def get(self, phase: "HookPhase | str", key: "Hashable") -> "Hook | None":
        phase = HookPhase(phase)
        return self._hooks.get(phase, {}).get(key)

    def iter_phase(self, phase: "HookPhase | str") -> "Iterator[Hook]":
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

        def sort_key(k: "Hashable") -> "tuple[int, int]":
            return (bucket[k].order, registration_order[k])

        result: "list[Hook]" = []
        visited: "set" = set()
        visiting: "set" = set()

        def visit(key: "Hashable") -> None:
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

    def validate(self, phase: "HookPhase | str | None" = None) -> "list[str]":
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

    def describe(self, phase: "HookPhase | str | None" = None) -> "list[dict[str, Any]]":
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

    def call(self, phase: "HookPhase | str", context: "Any" = None, reverse: bool = False) -> None:
        """Invoke every hook registered for ``phase``, in registry order.

        Always validates ``phase`` first: a missing *required* before/after
        reference or an ordering cycle must fail before any hook runs, not
        be silently skipped.

        A CONTEXT-invocation hook receives ``context`` (when one is given);
        a NO_ARGS (or opaque) hook is always called with no arguments.
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
