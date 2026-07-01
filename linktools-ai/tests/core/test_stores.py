"""Tests for the session store Protocols and file-backed defaults in
`linktools.ai.core.stores`, adapted from `sec-smartops-svc`'s
`tests/test_agent_session.py` (session-store coverage) to exercise `stores.py`
directly rather than the full `Session`/`Agent` stack.
"""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart

from linktools.ai.core.stores import (
    ArchiveArtifactStore,
    ArtifactStore,
    HistoryStore,
    InMemorySessionStatusStore,
    TranscriptStore,
    _estimate_model_messages,
)


def test_session_status_store_defaults_to_idle_then_tracks_updates():
    status_store = InMemorySessionStatusStore()

    initial = asyncio.run(status_store.get("session-1"))
    assert initial.type == "idle"
    assert initial.message is None

    async def _set_busy():
        from linktools.ai.core.stores import SessionStatusInfo
        await status_store.set(
            "session-1",
            SessionStatusInfo(type="busy", updated_at="2026-07-01T00:00:00+00:00"),
        )

    asyncio.run(_set_busy())
    current = asyncio.run(status_store.get("session-1"))

    assert current.type == "busy"
    assert current.message is None


def test_session_status_store_keeps_sessions_independent():
    status_store = InMemorySessionStatusStore()

    other = asyncio.run(status_store.get("session-2"))

    assert other.type == "idle"


def test_archive_artifact_store_wraps_archive_service(tmp_path):
    class StubArchiveService:
        def __init__(self):
            self.finalized = []
            self.restored = []

        async def finalize(self, trace_id: str):
            self.finalized.append(trace_id)
            return {"trace_id": trace_id}

        async def restore(self, trace_id: str):
            self.restored.append(trace_id)
            return tmp_path / trace_id

    service = StubArchiveService()
    store = ArchiveArtifactStore(service)
    session = SimpleNamespace(trace_id="TRC-1")

    finalized = asyncio.run(store.finalize(session))
    restored = asyncio.run(store.restore(session))

    assert finalized == {"trace_id": "TRC-1"}
    assert restored == tmp_path / "TRC-1"
    assert service.finalized == ["TRC-1"]
    assert service.restored == ["TRC-1"]


def test_archive_artifact_store_delegates_sidecar_persistence_to_fallback():
    calls = []

    class StubFallback:
        async def persist_call_sidecar(self, session, turn):
            calls.append((session, turn))

        async def finalize(self, session):
            return None

        async def restore(self, session):
            return None

    fallback = StubFallback()
    store = ArchiveArtifactStore(archive_service=SimpleNamespace(), fallback=fallback)
    session = SimpleNamespace(trace_id="TRC-2")
    turn = SimpleNamespace()

    asyncio.run(store.persist_call_sidecar(session, turn))

    assert calls == [(session, turn)]


def test_estimate_model_messages_is_zero_for_empty_history():
    assert _estimate_model_messages([]) == 0


def test_estimate_model_messages_is_positive_for_nonempty_history():
    messages = [ModelRequest(parts=[UserPromptPart(content="hello there")])]

    assert _estimate_model_messages(messages) > 0


def test_transcript_store_protocol_is_runtime_checkable():
    # Only `TranscriptStore` is declared with `@runtime_checkable`; `HistoryStore`
    # and `ArtifactStore` are plain `Protocol` (structural typing checked
    # statically, not via `isinstance`).
    class FakeTranscriptStore:
        async def head(self, session_id):
            raise NotImplementedError

        async def load(self, session_id, *, budget_tokens, after_seq=None, batch_size=64):
            raise NotImplementedError

        async def save(self, transcript):
            raise NotImplementedError

    assert isinstance(FakeTranscriptStore(), TranscriptStore)
    assert not isinstance(object(), TranscriptStore)


def test_history_store_and_artifact_store_are_plain_non_runtime_checkable_protocols():
    class FakeHistoryStore:
        async def load(self, session):
            raise NotImplementedError

        async def persist(self, session, turn):
            raise NotImplementedError

    class FakeArtifactStore:
        async def persist_call_sidecar(self, session, turn):
            raise NotImplementedError

        async def finalize(self, session):
            raise NotImplementedError

        async def restore(self, session):
            raise NotImplementedError

    # isinstance() against a non-@runtime_checkable Protocol raises TypeError;
    # this pins that these two Protocols were not upgraded silently.
    with pytest.raises(TypeError):
        isinstance(FakeHistoryStore(), HistoryStore)
    with pytest.raises(TypeError):
        isinstance(FakeArtifactStore(), ArtifactStore)
