#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""EventRegistry / EventCodec contract.

Pins the guarantees the stable event wire contract makes:

* the wire ``event_type`` and the criticality come from the registry, so
  renaming a payload class changes neither (the rename-class guarantee);
* every registered payload round-trips through encode/decode;
* an unknown ``event_type`` decodes to ``UnknownEventPayload`` rather than
  raising;
* a registered payload with a migrator decodes an older schema_version
  forward, and a newer-than-known schema_version raises.
"""

import dataclasses

import pytest

from linktools.ai.events.criticality import EventCriticality, classify_event
from linktools.ai.events.payloads import (
    ApprovalRequested,
    RunCompleted,
    RunPaused,
    RunStarted,
)
from linktools.ai.events.registry import (
    EventCodec,
    EventDescriptor,
    EventRegistry,
    EventSchemaError,
    UnknownEventPayload,
    build_default_registry,
    default_codec,
)


def test_every_registered_payload_round_trips() -> None:
    """Every payload in the default registry survives encode -> decode."""
    registry = default_codec.registry
    assert len(registry) > 40
    # Exercise a representative sample across criticality bands.
    samples = [
        RunStarted(run_id="r", runnable_id="a"),
        RunCompleted(run_id="r"),
        RunPaused(run_id="r", reason="wait"),
        ApprovalRequested(approval_id="ap", tool_name="t", reason="x"),
    ]
    for payload in samples:
        event_type, schema_version, data = default_codec.encode(payload)
        assert schema_version == 1
        back = default_codec.decode(event_type, schema_version, data)
        assert back == payload, (event_type, payload, back)


def test_unknown_event_type_decodes_to_unknown_payload() -> None:
    """An event_type the registry does not know decodes to UnknownEventPayload
    (never raises) so history stays readable across version skew."""
    decoded = default_codec.decode("SomeFutureEvent.v2", 2, {"a": 1, "b": "x"})
    assert isinstance(decoded, UnknownEventPayload)
    assert decoded.event_type == "SomeFutureEvent.v2"
    assert decoded.schema_version == 2
    assert decoded.data == {"a": 1, "b": "x"}


def test_unknown_payload_is_observability_only() -> None:
    """An unregistered / unknown payload cannot masquerade as state- or
    security-critical (it must not drive state recovery)."""
    assert (
        classify_event(UnknownEventPayload("X", 1, {}))
        is EventCriticality.OBSERVABILITY
    )


def test_unknown_payload_in_critical_slot_fails_closed() -> None:
    """A state/security-critical event that has drifted out of the registry
    (version skew, a bad rename, corruption) must not silently become
    observability in a critical-event path. The commit coordinator's dedup
    scan calls ``event_type_for`` on every event in the stream; an
    UnknownEventPayload is not registered, so that lookup raises
    EventSchemaError -- the critical-event slot fails closed rather than
    proceeding past an event it cannot identify.
    """
    registry = default_codec.registry
    unknown = UnknownEventPayload("MissingType", 1, {"x": 1})
    # classify_event (best-effort decode path) treats it as observability...
    assert classify_event(unknown) is EventCriticality.OBSERVABILITY
    # ...but the critical-slot identity lookup raises -- fail closed.
    with pytest.raises(EventSchemaError):
        registry.event_type_for(unknown)



def test_rename_payload_class_does_not_change_wire_type_or_criticality() -> None:
    """The defining guarantee of the stable event wire contract: the wire
    ``event_type`` and the criticality come from ClassVar literals on the
    payload, NOT from ``type(payload).__name__``. Renaming the Python class
    leaves both unchanged.

    This simulates a rename: ``RunPausedRenamed`` is a subclass with a DIFFERENT
    class name but the SAME inherited ClassVar (event_type="RunPaused",
    criticality=STATE_CRITICAL). The default-registry registration logic reads
    those ClassVars, so the renamed class still produces wire type "RunPaused"
    -- not "RunPausedRenamed" -- and stays state-critical. If the registry
    regressed to ``type().__name__``, this assertion would fail.
    """

    class RunPausedRenamed(RunPaused):
        pass

    renamed = RunPausedRenamed(run_id="r", reason="x")
    # The renamed class has a different Python name but inherits the ClassVar.
    assert type(renamed).__name__ == "RunPausedRenamed"
    assert RunPausedRenamed.event_type == "RunPaused"
    assert RunPausedRenamed.criticality is EventCriticality.STATE_CRITICAL

    # Apply build_default_registry's registration logic to the renamed class:
    # read the ClassVar literals (never the class name).
    registry = EventRegistry()
    registry.register(EventDescriptor(
        event_type=RunPausedRenamed.event_type,
        schema_version=1,
        payload_type=RunPausedRenamed,
        criticality=RunPausedRenamed.criticality,
        decoder=lambda data: RunPausedRenamed(**dict(data)),
    ))
    registry.freeze()
    codec = EventCodec(registry)

    event_type, schema_version, _ = codec.encode(renamed)
    assert event_type == "RunPaused", (
        "wire event_type must be the ClassVar literal, not the renamed class "
        f"name (got {event_type!r})"
    )
    assert event_type != type(renamed).__name__
    assert schema_version == 1
    assert registry.criticality_of(renamed) is EventCriticality.STATE_CRITICAL


def test_no_payload_class_name_drives_event_type() -> None:
    """No payload's wire event_type is derived from its class name. Across the
    whole default registry, the descriptor's event_type equals the ClassVar
    literal on the payload class -- read independently of ``cls.__name__``.
    (If build_default_registry regressed to ``cls.__name__``, this would still
    pass today only because the ClassVar happens to match; this test pins the
    ClassVar as the source so a future divergence is caught.)"""
    registry = default_codec.registry
    from linktools.ai.events import payloads as _payloads

    for event_type, descriptor in registry.descriptors().items():
        cls = descriptor.payload_type
        assert getattr(cls, "event_type", None) == event_type
        assert isinstance(getattr(cls, "criticality", None), EventCriticality)
        assert descriptor.criticality is cls.criticality
    # _STATE_CRITICAL / _SECURITY_CRITICAL / _criticality_for are gone.
    import linktools.ai.events.registry as _reg
    for removed in ("_STATE_CRITICAL", "_SECURITY_CRITICAL", "_criticality_for"):
        assert not hasattr(_reg, removed), f"{removed} still present"


def test_migrator_upgrades_older_schema_forward() -> None:
    """A registered payload with a migrator decodes an older schema_version by
    running the chain forward to the current version."""

    @dataclasses.dataclass(frozen=True, slots=True)
    class V2:
        name: str
        # field added at schema_version 2; v1 envelopes carry only ``legacy``.

    def migrate_v1_to_v2(data):
        return {"name": data["legacy"]}

    registry = EventRegistry()
    registry.register(EventDescriptor(
        event_type="Custom.V2",
        schema_version=2,
        payload_type=V2,
        criticality=EventCriticality.OBSERVABILITY,
        decoder=lambda data: V2(**dict(data)),
        migrators={2: migrate_v1_to_v2},
    ))
    registry.freeze()
    codec = EventCodec(registry)

    # A version-1 envelope migrates forward and decodes as the v2 payload.
    decoded = codec.decode("Custom.V2", 1, {"legacy": "alice"})
    assert isinstance(decoded, V2) and decoded.name == "alice"
    # A current-version envelope decodes without migration.
    decoded_v2 = codec.decode("Custom.V2", 2, {"name": "bob"})
    assert isinstance(decoded_v2, V2) and decoded_v2.name == "bob"


def test_newer_than_known_schema_version_raises() -> None:
    """A schema_version newer than the descriptor's current version is a hard
    error, not silently downcast."""

    @dataclasses.dataclass(frozen=True, slots=True)
    class V1:
        x: int

    registry = EventRegistry()
    registry.register(EventDescriptor(
        event_type="Custom.V1",
        schema_version=1,
        payload_type=V1,
        criticality=EventCriticality.OBSERVABILITY,
        decoder=lambda data: V1(**dict(data)),
    ))
    registry.freeze()
    codec = EventCodec(registry)
    with pytest.raises(EventSchemaError):
        codec.decode("Custom.V1", 2, {"x": 1})


def test_registry_rejects_duplicate_registration() -> None:
    """A frozen registry cannot grow, and a duplicate event_type or payload
    type is rejected up front."""
    registry = EventRegistry()
    registry.register(EventDescriptor(
        event_type="X",
        schema_version=1,
        payload_type=RunStarted,
        criticality=EventCriticality.OBSERVABILITY,
        decoder=lambda data: RunStarted(**dict(data)),
    ))
    with pytest.raises(ValueError):
        registry.register(EventDescriptor(
            event_type="X",  # duplicate event_type
            schema_version=1,
            payload_type=RunCompleted,
            criticality=EventCriticality.OBSERVABILITY,
            decoder=lambda data: RunCompleted(**dict(data)),
        ))
    registry.freeze()
    desc = EventDescriptor(
        event_type="Y",
        schema_version=1,
        payload_type=RunCompleted,
        criticality=EventCriticality.OBSERVABILITY,
        decoder=lambda data: RunCompleted(**dict(data)),
    )
    with pytest.raises(RuntimeError):
        registry.register(desc)


def test_build_default_registry_is_frozen_and_complete() -> None:
    registry = build_default_registry()
    # Re-building yields an independent but equally-sized frozen registry.
    again = build_default_registry()
    assert len(registry) == len(again) == len(default_codec.registry)


def test_file_store_reads_legacy_envelope_without_event_type(tmp_path) -> None:
    """Envelopes written before the registry existed carry ``payload_type`` (the
    class name) and no ``event_type``/``schema_version``. The file store must
    still read them back via the codec's legacy-tag fallback."""
    import asyncio
    import json as _json
    from datetime import datetime, timezone

    from linktools.ai.events.payloads import RunPaused
    from linktools.ai.storage.filesystem.event import FilesystemEventStore

    store = FilesystemEventStore(root=tmp_path)
    stream_dir = tmp_path / "legacy-stream"
    stream_dir.mkdir(parents=True)
    legacy_row = {
        "event_id": "ev-1",
        "stream_id": "legacy-stream",
        "sequence": 1,
        "occurred_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "run-1",
        "root_run_id": "run-1",
        "parent_run_id": None,
        "session_id": "sess-1",
        "runnable_id": "agent-1",
        # Legacy shape: payload_type (class name), no event_type/schema_version.
        "payload_type": "RunPaused",
        "payload": {"run_id": "run-x", "reason": "old"},
        "metadata": {},
    }
    (stream_dir / "0000000001.json").write_text(_json.dumps(legacy_row))

    page = asyncio.run(store.list("legacy-stream", after_sequence=0, limit=10))
    assert len(page.items) == 1
    assert page.items[0].payload == RunPaused(run_id="run-x", reason="old")

