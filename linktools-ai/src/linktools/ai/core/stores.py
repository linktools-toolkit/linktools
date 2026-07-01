from __future__ import annotations

"""Session store abstractions: TranscriptStore / HistoryStore / ArtifactStore /
SessionStatusStore Protocols and their concrete implementations
(FileHistoryStore, DbHistoryStore, LocalArtifactStore, ReadOnlyArtifactStore,
ArchiveArtifactStore, InMemorySessionStatusStore)."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from .model_runtime import (
    CallPromptSnapshot,
    SessionContextSnapshot,
    _model_message_to_dicts,
    load_message_history,
    new_call_messages,
    write_call_prompt,
    write_session_context,
)
from .session_window import (
    SessionSummary,
    constrain_summary,
    estimate_summary_tokens,
    prepend_summary_message,
    summary_prompt_budget,
    strip_summary_message,
)

if TYPE_CHECKING:
    # `.session` imports from this module at runtime (FileHistoryStore, DbHistoryStore,
    # etc. are used as dataclass field defaults), so importing it back here at module
    # load time would be circular. These names are only needed for static typing;
    # runtime call sites that need the real class (DbSession, SessionEvent,
    # SessionTranscriptWrite, SessionTranscriptHead) import it locally instead.
    from .session import (
        Session,
        SessionEvent,
        SessionTranscript,
        SessionTranscriptHead,
        SessionTranscriptWrite,
        SessionTurn,
    )


@runtime_checkable
class TranscriptStore(Protocol):
    """Durable backend for a DB-backed session history store.

    This is the DB analog of `context.json`, declared here so the agent layer stays
    decoupled from secops/DB internals.
    """

    async def head(self, session_id: str) -> SessionTranscriptHead:
        """Return the stored session head for `session_id` (or an empty head)."""
        ...

    async def load(
        self,
        session_id: str,
        *,
        budget_tokens: int,
        after_seq: int | None = None,
        batch_size: int = 64,
    ) -> SessionTranscript:
        """Return the restore snapshot plus an incrementally-read tail event window."""
        ...

    async def save(self, transcript: SessionTranscriptWrite) -> None:
        """Upsert the session head and append any new ordered events for `session_id`."""
        ...


class HistoryStore(Protocol):
    """Persistence backend for multi-turn session history."""

    async def load(self, session: "Session") -> list[ModelMessage]:
        ...

    async def persist(self, session: "Session", turn: SessionTurn) -> None: ...


class ArtifactStore(Protocol):
    """Persistence backend for non-history session artifacts."""

    async def persist_call_sidecar(self, session: "Session", turn: SessionTurn) -> None: ...

    async def finalize(self, session: "Session") -> dict[str, Any] | None:
        ...

    async def restore(self, session: "Session") -> Path | None:
        ...


SessionStatus = Literal["idle", "busy", "retry", "error"]


@dataclass(frozen=True, slots=True)
class SessionStatusInfo:
    type: SessionStatus
    updated_at: str
    message: str | None = None


class SessionStatusStore(Protocol):
    """Runtime-only session activity state, separate from durable session metadata."""

    async def get(self, session_id: str) -> SessionStatusInfo:
        ...

    async def set(self, session_id: str, status: SessionStatusInfo) -> None:
        ...


class InMemorySessionStatusStore:
    def __init__(self) -> None:
        self._states: dict[str, SessionStatusInfo] = {}

    async def get(self, session_id: str) -> SessionStatusInfo:
        return self._states.get(
            session_id,
            SessionStatusInfo(type="idle", updated_at=datetime.now(timezone.utc).isoformat()),
        )

    async def set(self, session_id: str, status: SessionStatusInfo) -> None:
        self._states[session_id] = status


class FileHistoryStore:
    async def load(self, session: "Session") -> list[ModelMessage]:
        return await asyncio.to_thread(load_message_history, session.session_dir)

    async def persist(self, session: "Session", turn: SessionTurn) -> None:
        await asyncio.to_thread(
            write_session_context,
            session.session_dir,
            SessionContextSnapshot(
                trace_id=session.trace_id,
                session_id=session.session_id,
                messages=turn.all_messages,
                model=turn.model,
                token_usage=turn.token_usage,
                llm_call=turn.llm_call,
            ),
        )


def _events_to_raw_messages(events: list[SessionEvent]) -> list[Any]:
    result = []
    for event in events:
        message = (event.payload or {}).get("message")
        if message:
            result.append(message)
    return result


def _load_session_summary(meta_json: dict[str, Any] | None) -> SessionSummary | None:
    if not isinstance(meta_json, dict):
        return None
    return SessionSummary.from_dict(meta_json.get("summary") if isinstance(meta_json.get("summary"), dict) else None)


def _unsummarized_trimmed_messages(
    trimmed_messages: list[ModelMessage],
    *,
    existing_summary: SessionSummary | None,
    covered_until_seq: int,
) -> list[ModelMessage]:
    if not trimmed_messages:
        return []
    already_covered_until = existing_summary.covered_until_seq if existing_summary is not None else 0
    unsummarized_count = covered_until_seq - already_covered_until
    if unsummarized_count <= 0:
        return []
    if unsummarized_count >= len(trimmed_messages):
        return list(trimmed_messages)
    return list(trimmed_messages[-unsummarized_count:])


class DbHistoryStore:
    def __init__(self, store: TranscriptStore) -> None:
        self._store = store

    async def load(self, session: "Session") -> list[ModelMessage]:
        from .session import DbSession

        if isinstance(session, DbSession) and session.loaded_head_seq > 0 and session.loaded_messages:
            prompt_summary = constrain_summary(
                session.loaded_summary,
                max_tokens=summary_prompt_budget(session.history_token_budget),
            )
            remaining_budget = max(
                0,
                session.history_token_budget
                - estimate_summary_tokens(prompt_summary)
                - _estimate_model_messages(session.loaded_messages),
            )
            transcript = await self._store.load(
                session.session_id,
                budget_tokens=remaining_budget,
                after_seq=session.loaded_head_seq,
            )
            raw = _events_to_raw_messages(transcript.events)
            if raw:
                session.loaded_messages.extend(ModelMessagesTypeAdapter.validate_python(raw))
                session.loaded_head_seq = transcript.loaded_until_seq or session.loaded_head_seq
            return prepend_summary_message(session.loaded_messages, prompt_summary)

        transcript = await self._store.load(
            session.session_id,
            budget_tokens=getattr(session, "history_token_budget", 24000),
        )
        effective_budget = _resolved_history_token_budget(session, transcript.head)
        raw = list(transcript.head.snapshot_messages) + _events_to_raw_messages(transcript.events)
        summary = _load_session_summary(transcript.head.meta_json)
        prompt_summary = constrain_summary(
            summary,
            max_tokens=summary_prompt_budget(effective_budget),
        )
        if not raw:
            if isinstance(session, DbSession):
                session.history_token_budget = effective_budget
                session.loaded_head_seq = transcript.loaded_until_seq or transcript.head.head_seq or transcript.head.snapshot_seq
                session.loaded_messages = []
                session.loaded_summary = summary
            return prepend_summary_message([], prompt_summary)
        messages = list(ModelMessagesTypeAdapter.validate_python(raw))
        window = session.window_policy.build(
            messages,
            budget_tokens=effective_budget,
            head_seq=transcript.loaded_until_seq or transcript.head.head_seq or transcript.head.snapshot_seq,
            summary=summary,
        )
        if isinstance(session, DbSession):
            session.history_token_budget = effective_budget
            session.loaded_head_seq = transcript.loaded_until_seq or transcript.head.head_seq or transcript.head.snapshot_seq
            session.loaded_messages = list(window.recent_messages)
            session.loaded_summary = summary
        return prepend_summary_message(window.recent_messages, prompt_summary)

    async def persist(self, session: "Session", turn: SessionTurn) -> None:
        from .session import DbSession, SessionTranscriptHead, SessionTranscriptWrite

        prior_head_seq = getattr(session, "loaded_head_seq", 0)
        history_token_budget = getattr(session, "history_token_budget", 24000)
        existing_summary = getattr(session, "loaded_summary", None)
        prompt_summary = constrain_summary(
            existing_summary,
            max_tokens=summary_prompt_budget(history_token_budget),
        )
        raw_history = strip_summary_message(turn.history, prompt_summary)
        raw_all_messages = strip_summary_message(turn.all_messages, prompt_summary)
        head_seq = prior_head_seq + max(0, len(raw_all_messages) - len(raw_history))
        window = session.window_policy.build(
            raw_all_messages,
            budget_tokens=history_token_budget,
            head_seq=head_seq,
            summary=existing_summary,
        )
        trimmed_for_summary = _unsummarized_trimmed_messages(
            window.trimmed_messages,
            existing_summary=window.summary,
            covered_until_seq=window.snapshot_boundary_seq,
        )
        summary = session.summary_policy.summarize(
            trimmed_for_summary,
            existing_summary=window.summary,
            covered_until_seq=window.snapshot_boundary_seq,
        )
        if summary is None:
            summary = window.summary
        summary = constrain_summary(summary)
        snapshot_messages = ModelMessagesTypeAdapter.dump_python(window.recent_messages, mode="json")
        snapshot_boundary_seq = window.snapshot_boundary_seq
        snapshot_token_estimate = _estimate_dumped_tokens(snapshot_messages)
        meta = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "current_model": {"model_type": turn.model.model_type, "model": turn.model.model},
            "token_usage": turn.token_usage,
            "last_call": turn.llm_call,
        }
        if summary is not None:
            meta["summary"] = summary.as_dict()
        request_id = getattr(session, "pending_request_id", None)
        events = _build_session_events(
            raw_all_messages[len(raw_history):],
            start_seq=prior_head_seq + 1,
            session_id=session.session_id,
            turn_key=str(turn.llm_call.get("call_id") or f"turn-{prior_head_seq + 1}"),
            call_id=str(turn.llm_call.get("call_id") or "") or None,
            seed_display_entries=turn.display_entries,
            request_id=request_id,
        )
        if request_id is not None and isinstance(session, DbSession):
            session.pending_request_id = None
        head_seq = prior_head_seq + len(events)
        await self._store.save(
            SessionTranscriptWrite(
                session_id=session.session_id,
                trace_id=session.trace_id,
                events=events,
                head=SessionTranscriptHead(
                    trace_id=session.trace_id,
                    session_kind=session.session_dir.name,
                    parent_session_id=session.parent_session_id,
                    head_seq=head_seq,
                    snapshot_seq=snapshot_boundary_seq,
                    snapshot_messages=snapshot_messages,
                    snapshot_token_estimate=snapshot_token_estimate,
                    history_token_budget=history_token_budget,
                    status="active",
                    meta_json=meta,
                ),
            )
        )
        if isinstance(session, DbSession):
            session.loaded_head_seq = head_seq
            session.loaded_messages = list(window.recent_messages)
            session.history_token_budget = history_token_budget
            session.loaded_summary = summary


def _build_session_events(
    messages: list[ModelMessage],
    *,
    start_seq: int,
    session_id: str,
    turn_key: str,
    call_id: str | None,
    seed_display_entries: list[dict[str, Any]] | None = None,
    request_id: str | None = None,
) -> list[SessionEvent]:
    from .session import SessionEvent

    events: list[SessionEvent] = []
    dumped_messages = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    known_tool_names: dict[str, str] = {}
    for offset, (message, dumped) in enumerate(zip(messages, dumped_messages, strict=False)):
        entries = _model_message_to_dicts(message)
        primary = next((entry for entry in entries if entry.get("role") in {"user", "assistant"}), None)
        primary = primary or next((entry for entry in entries if entry.get("role")), None) or {}
        role = str(primary.get("role") or "system")
        content = "\n".join(
            str(entry.get("content") or "")
            for entry in entries
            if entry.get("role") == role and entry.get("content")
        ).strip()
        payload: dict[str, Any] = {
            "message": dumped,
            "entries": entries,
            "display_entries": _build_display_entries(
                entries,
                known_tool_names=known_tool_names,
                seed_display_entries=seed_display_entries if role == "user" and offset == 0 else None,
            ),
        }
        if seed_display_entries and role == "user" and offset == 0:
            payload["content"] = _primary_display_text(seed_display_entries)
        elif content:
            payload["content"] = content
        token_estimate = _estimate_dumped_tokens([dumped])
        event_id = (
            f"{session_id}:req:{request_id}" if (request_id and offset == 0)
            else f"{session_id}:{turn_key}:{offset}"
        )
        events.append(
            SessionEvent(
                seq=start_seq + offset,
                event_id=event_id,
                event_type=f"{role}_message",
                role=role,
                turn_id=None,
                call_id=call_id,
                parent_call_id=None,
                visible_in_ui=role in {"user", "assistant"} and bool(payload.get("content")),
                token_estimate=token_estimate,
                payload=payload,
            )
        )
    return events


def _build_display_entries(
    entries: list[dict[str, Any]],
    *,
    known_tool_names: dict[str, str],
    seed_display_entries: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    from .session import normalize_display_entry, normalize_display_entries

    display_entries: list[dict[str, Any]] = []
    if seed_display_entries:
        return normalize_display_entries(seed_display_entries)
    for entry in entries:
        role = str(entry.get("role") or "")
        if role == "user":
            text = str(entry.get("content") or "").strip()
            if text:
                display_entries.append(normalize_display_entry({"kind": "user_text", "role": "user", "text": text}))
        elif role == "assistant":
            for call in entry.get("tool_calls") or []:
                function = call.get("function") if isinstance(call, dict) else {}
                tool_call_id = str(call.get("id") or "")
                name = str((function or {}).get("name") or "")
                if tool_call_id and name:
                    known_tool_names[tool_call_id] = name
                if name:
                    display_entries.append(
                        normalize_display_entry(
                            {
                                "kind": "tool_call",
                                "role": "assistant",
                                "name": name,
                                "tool_call_id": tool_call_id or None,
                                "state": "completed",
                            }
                        )
                    )
            text = str(entry.get("content") or "").strip()
            if text:
                display_entries.append(
                    normalize_display_entry({"kind": "assistant_text", "role": "assistant", "text": text})
                )
        elif role == "tool":
            text = str(entry.get("content") or "").strip()
            tool_call_id = str(entry.get("tool_call_id") or "")
            name = known_tool_names.get(tool_call_id) or tool_call_id
            if text:
                display_entries.append(
                    normalize_display_entry(
                        {
                            "kind": "tool_result",
                            "role": "tool",
                            "name": name or "tool",
                            "tool_call_id": tool_call_id or None,
                            "text": text,
                        }
                    )
                )
    return display_entries


def _primary_display_text(display_entries: list[dict[str, Any]]) -> str:
    for item in display_entries:
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return text
    return ""




def _estimate_dumped_tokens(payload: Any) -> int:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


def _estimate_model_messages(messages: list[ModelMessage]) -> int:
    if not messages:
        return 0
    return _estimate_dumped_tokens(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))


def _resolved_history_token_budget(session: "Session", head: SessionTranscriptHead) -> int:
    configured_budget = getattr(session, "history_token_budget", 24000)
    if head.is_empty:
        return configured_budget
    stored_budget = int(head.history_token_budget or configured_budget)
    return min(configured_budget, stored_budget)


class LocalArtifactStore:
    async def persist_call_sidecar(self, session: "Session", turn: SessionTurn) -> None:
        await asyncio.to_thread(
            write_call_prompt,
            session.session_dir,
            CallPromptSnapshot(
                call_id=str(turn.llm_call.get("call_id") or ""),
                messages=new_call_messages(turn.history, turn.all_messages, system_prompt=turn.system_prompt),
            ),
        )

    async def finalize(self, session: "Session") -> dict[str, Any] | None:
        del session
        return None

    async def restore(self, session: "Session") -> Path | None:
        return session.trace_root


class ReadOnlyArtifactStore:
    """ArtifactStore that makes no writes — use when a session reads an existing trace."""

    async def persist_call_sidecar(self, session: "Session", turn: SessionTurn) -> None:
        pass

    async def finalize(self, session: "Session") -> dict[str, Any] | None:
        del session
        return None

    async def restore(self, session: "Session") -> "Path | None":
        return session.trace_root


class ArchiveArtifactStore:
    def __init__(self, archive_service: Any, *, fallback: ArtifactStore | None = None) -> None:
        self._archive_service = archive_service
        self._fallback = fallback or LocalArtifactStore()

    async def persist_call_sidecar(self, session: "Session", turn: SessionTurn) -> None:
        await self._fallback.persist_call_sidecar(session, turn)

    async def finalize(self, session: "Session") -> dict[str, Any] | None:
        return await self._archive_service.finalize(session.trace_id)

    async def restore(self, session: "Session") -> Path | None:
        return await self._archive_service.restore(session.trace_id)


