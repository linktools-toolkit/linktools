#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stable event wire contract: EventDescriptor / EventRegistry / EventCodec.

Every event payload is described by an ``EventDescriptor`` keyed on a *stable*
``event_type`` string -- not on the Python payload class name. Renaming a
payload class therefore never changes the wire event type, the schema migration
path, or the criticality; those live here, in the registry.

The codec is the single encode/decode path used by both the Filesystem and the
SQLAlchemy event stores. Unknown event types decode to ``UnknownEventPayload``
(best-effort, never state-recoverable); versioned payloads migrate forward
through chained migrators.

For wire compatibility with envelopes written before the registry existed, each
payload's ``event_type`` ClassVar literal is initialized to its class name and
``schema_version`` is 1. The literal is decoupled from the live class name --
it is a frozen constant on the payload -- so renaming a payload class cannot
change the wire event type or the criticality.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Mapping, TypeVar

from .criticality import EventCriticality
from .payloads import EventPayload

T = TypeVar("T")

# A migrator upgrades a payload dict from ``schema_version - 1`` to
# ``schema_version``. Chained migrators carry a payload from any older version
# to the current one. Returning a dict keeps migrators pure (no dataclass
# reconstruction mid-chain).
EventMigrator = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class EventDescriptor(Generic[T]):
    """Stable description of one event payload type."""

    event_type: str
    schema_version: int
    payload_type: "type[T]"
    criticality: EventCriticality
    decoder: "Callable[[Mapping[str, Any]], T]"
    migrators: "Mapping[int, EventMigrator]" = field(default_factory=dict)

    def decode(self, data: "Mapping[str, Any]", schema_version: "int | None") -> T:
        """Reconstruct the payload, migrating forward from ``schema_version``."""
        current = self.schema_version
        from_version = current if schema_version is None else schema_version
        if from_version > current:
            raise EventSchemaError(
                f"event {self.event_type!r} schema_version {from_version} is "
                f"newer than the registered version {current}"
            )
        migrated = dict(data)
        # Apply migrators in order: each migrator at version N upgrades a
        # version-(N-1) payload to version N.
        for target in range(from_version + 1, current + 1):
            migrator = self.migrators.get(target)
            if migrator is None:
                raise EventSchemaError(
                    f"event {self.event_type!r} has no migrator to schema "
                    f"version {target} (from {from_version})"
                )
            migrated = dict(migrator(migrated))
        return self.decoder(migrated)


@dataclass(frozen=True, slots=True)
class UnknownEventPayload:
    """Stands in for an event whose ``event_type`` is not registered.

    Observability-only: it preserves the original type tag and fields for
    audit/diagnostics but must never drive state recovery. A state- or
    security-critical event that decodes to this is a version-skew bug; the
    commit path treats UnknownEventPayload in a critical slot as a hard error.
    """

    event_type: str
    schema_version: "int | None"
    data: "Mapping[str, Any]"


class EventSchemaError(Exception):
    """Raised when a registered event cannot be migrated to its current
    schema (a missing intermediate migrator). Distinct from an *unknown* event
    type, which decodes to UnknownEventPayload rather than raising."""


class EventRegistry:
    """Frozen map of ``event_type`` -> ``EventDescriptor`` plus the reverse
    ``payload_type`` -> ``event_type`` lookup the codec uses to encode.

    Built once (at Runtime construction per §6.4: "EventCodec registry 在
    Runtime 构造时冻结"); mutation after ``freeze()`` raises.
    """

    def __init__(self) -> None:
        self._by_type: "dict[str, EventDescriptor]" = {}
        self._by_payload: "dict[type, str]" = {}
        self._frozen = False

    def register(self, descriptor: "EventDescriptor") -> None:
        if self._frozen:
            raise RuntimeError("EventRegistry is frozen; cannot register")
        if descriptor.event_type in self._by_type:
            raise ValueError(
                f"event_type {descriptor.event_type!r} already registered"
            )
        if descriptor.payload_type in self._by_payload:
            raise ValueError(
                f"payload type {descriptor.payload_type!r} already registered"
            )
        self._by_type[descriptor.event_type] = descriptor
        self._by_payload[descriptor.payload_type] = descriptor.event_type

    def freeze(self) -> None:
        self._frozen = True

    def __len__(self) -> int:
        return len(self._by_type)

    def __contains__(self, event_type: object) -> bool:
        return event_type in self._by_type

    def descriptor_for(self, event_type: str) -> "EventDescriptor | None":
        return self._by_type.get(event_type)

    def descriptors(self) -> "Mapping[str, EventDescriptor]":
        """A read-only view of every registered event_type -> descriptor."""
        return dict(self._by_type)

    def event_type_for(self, payload: EventPayload) -> str:
        """The stable wire type for ``payload``. Raises if the payload's class
        was never registered -- encoding an unregistered payload is a bug."""
        try:
            return self._by_payload[type(payload)]
        except KeyError:
            raise EventSchemaError(
                f"payload type {type(payload).__name__!r} is not registered"
            ) from None

    def criticality_of(self, payload: EventPayload) -> EventCriticality:
        """The single source of criticality (plan §4.5)."""
        event_type = self._by_payload.get(type(payload))
        if event_type is None:
            return EventCriticality.OBSERVABILITY
        return self._by_type[event_type].criticality


class EventCodec:
    """Encode/decode envelopes against a frozen :class:`EventRegistry`."""

    def __init__(
        self,
        registry: EventRegistry,
        *,
        metrics: "Any | None" = None,
    ) -> None:
        self._registry = registry
        # Optional ObservabilityMetrics sink: when wired, codec failures
        # increment ``event_codec_failure_total`` (attribute ``phase`` is
        # ``encode``/``decode``/``migrate``). Default None = no-op, so
        # existing callers that construct ``EventCodec(registry)`` and the
        # module-level ``default_codec`` keep their no-metrics behavior.
        self._metrics = metrics

    @property
    def registry(self) -> EventRegistry:
        return self._registry

    def encode(self, payload: EventPayload) -> "tuple[str, int, dict[str, Any]]":
        """Return ``(event_type, schema_version, data)`` for ``payload``."""
        try:
            event_type = self._registry.event_type_for(payload)
        except EventSchemaError:
            if self._metrics is not None:
                self._metrics.counter(
                    "event_codec_failure_total", attributes={"phase": "encode"}
                )
            raise
        descriptor = self._registry.descriptor_for(event_type)
        assert descriptor is not None  # event_type_for guarantees it
        data = _payload_to_mapping(payload)
        return event_type, descriptor.schema_version, data

    def decode(
        self,
        event_type: str,
        schema_version: "int | None",
        data: "Mapping[str, Any]",
    ) -> EventPayload:
        """Reconstruct a payload. Unknown types decode to
        :class:`UnknownEventPayload` (never raise)."""
        descriptor = self._registry.descriptor_for(event_type)
        if descriptor is None:
            return UnknownEventPayload(
                event_type=event_type,
                schema_version=schema_version,
                data=dict(data),
            )
        try:
            return descriptor.decode(data, schema_version)
        except EventSchemaError:
            # A migrate failure (missing migrator or a future schema_version)
            # surfaces here. ``phase`` distinguishes a migration-time failure
            # from an encode-side one.
            if self._metrics is not None:
                self._metrics.counter(
                    "event_codec_failure_total", attributes={"phase": "migrate"}
                )
            raise


def _payload_to_mapping(payload: EventPayload) -> "dict[str, Any]":
    """Structural payload -> dict, independent of the codec the store uses for
    envelope-level persistence (kept here so encode/decode are symmetric)."""
    import dataclasses as _dc

    return {
        f.name: _to_plain(getattr(payload, f.name))
        for f in _dc.fields(payload)
    }


def _to_plain(value: Any) -> Any:
    import dataclasses as _dc

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if _dc.is_dataclass(value):
        return {f.name: _to_plain(getattr(value, f.name)) for f in _dc.fields(value)}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _to_plain(v) for k, v in value.items()}
    return value


__all__: "list[str]" = [
    "EventCodec",
    "EventDescriptor",
    "EventMigrator",
    "EventRegistry",
    "EventSchemaError",
    "UnknownEventPayload",
    "build_default_registry",
    "default_codec",
]


# --- default registry -------------------------------------------------------
# Each payload class declares its own ``event_type`` and ``criticality`` as
# ClassVar literals (see payloads.py). The registry reads those constants -- it
# never derives either from the payload's class name. Renaming a payload class
# therefore cannot change the wire event type or the criticality: the literals
# travel with the class. This is the defining guarantee of the stable event
# wire contract.


def build_default_registry() -> EventRegistry:
    """Register every payload in :mod:`linktools.ai.events.payloads`.

    Each payload carries its own ``event_type`` and ``criticality`` ClassVar
    literals; the registry copies them onto the descriptor. ``schema_version``
    is 1 for every payload. The returned registry is frozen.
    """
    import dataclasses as _dc

    from . import payloads as _payloads

    registry = EventRegistry()
    missing: "list[str]" = []
    for name in dir(_payloads):
        cls = getattr(_payloads, name)
        if not (isinstance(cls, type) and _dc.is_dataclass(cls)
                and cls.__module__ == _payloads.__name__):
            continue
        # UnknownEventPayload is the codec's own fallback, not a registered wire
        # payload.
        if cls is UnknownEventPayload:
            continue
        event_type = getattr(cls, "event_type", None)
        criticality = getattr(cls, "criticality", None)
        if not isinstance(event_type, str) or not isinstance(criticality, EventCriticality):
            missing.append(cls.__name__)
            continue
        registry.register(EventDescriptor(
            event_type=event_type,
            schema_version=1,
            payload_type=cls,
            criticality=criticality,
            decoder=_make_decoder(cls),
            migrators={},
        ))
    if missing:
        raise EventSchemaError(
            "payload classes missing event_type/criticality ClassVar literals: "
            + ", ".join(sorted(missing))
        )
    registry.freeze()
    return registry


def _make_decoder(cls: type) -> "Callable[[Mapping[str, Any]], object]":
    def _decode(data: "Mapping[str, Any]", _cls: type = cls):
        return _cls(**dict(data))
    return _decode


default_codec = EventCodec(build_default_registry())

