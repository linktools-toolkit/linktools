import dataclasses
from pathlib import Path
from typing import Any, Mapping

import pytest

from linktools.ai.events.payloads import (
    RunCompleted,
    SecurityDegraded,
    TruncatedSecurityEvent,
)
from linktools.ai.security.emitter import (
    DefaultSecurityEventSanitizer,
    EventStoreSecurityEventEmitter,
)
from linktools.ai.storage.file.event import FileEventStore


def _oversized_mapping() -> "Mapping[str, Any]":
    # Each value stays well under the per-field text limit (4096), so the only
    # way the serialized form exceeds the payload budget is aggregate size --
    # exactly the case the truncation path must handle.
    return {f"key_{i}": "value_" * 64 for i in range(2000)}


def test_oversized_event_sanitizes_to_dataclass_not_dict():
    sanitizer = DefaultSecurityEventSanitizer()
    event = RunCompleted(run_id="run-1", result_summary=_oversized_mapping())

    sanitized = sanitizer.sanitize(event)

    assert dataclasses.is_dataclass(sanitized)
    assert not isinstance(sanitized, dict)
    assert isinstance(sanitized, TruncatedSecurityEvent)
    assert sanitized.original_event_type == "RunCompleted"
    assert sanitized.reason == "payload_too_large"
    assert sanitized.original_size_bytes > sanitizer._MAX_PAYLOAD
    # The oversized payload must not survive into the persisted surrogate.
    assert "value_" not in str(sanitized)


@pytest.mark.asyncio
async def test_oversized_security_event_persists_via_file_store_fail_closed(tmp_path: Path):
    store = FileEventStore(root=tmp_path)
    emitter = EventStoreSecurityEventEmitter(
        store, context=_ctx(), failure_mode="fail_closed")

    # Would raise TypeError -> ToolSecurityAuditError before the fix, because the
    # sanitizer returned a plain dict that FileEventStore.asdict() cannot handle.
    await emitter.emit_security(
        RunCompleted(run_id="run-1", result_summary=_oversized_mapping()))

    page = await store.list("run-1")
    assert len(page.items) == 1
    persisted = page.items[0].payload
    assert isinstance(persisted, TruncatedSecurityEvent)
    assert persisted.original_event_type == "RunCompleted"


@pytest.mark.asyncio
async def test_oversized_security_event_persists_best_effort(tmp_path: Path):
    store = FileEventStore(root=tmp_path)
    emitter = EventStoreSecurityEventEmitter(
        store, context=_ctx(), failure_mode="best_effort")

    await emitter.emit_security(
        RunCompleted(run_id="run-1", result_summary=_oversized_mapping()))

    page = await store.list("run-1")
    # best_effort must not silently drop a valid (truncated) audit event.
    assert len(page.items) == 1
    assert isinstance(page.items[0].payload, TruncatedSecurityEvent)


def test_normal_sized_event_passes_through_unchanged():
    sanitizer = DefaultSecurityEventSanitizer()
    event = SecurityDegraded(run_id="run-1", component="mcp-discovery", reason="ok")

    sanitized = sanitizer.sanitize(event)

    assert type(sanitized) is SecurityDegraded
    assert sanitized.reason == "ok"


@dataclasses.dataclass
class _Ctx:
    run_id: str = "run-1"
    root_run_id: str = "run-1"
    parent_run_id: "str | None" = None
    session_id: str = "run-1"
    runnable_id: str = "agent"


def _ctx() -> _Ctx:
    return _Ctx()
