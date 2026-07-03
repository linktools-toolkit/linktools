#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RemoteHistoryStore: session history backed by any TranscriptStore implementation."""

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter

from .history import _model_message_to_dicts
from .protocols import TranscriptStore
from .window import (
    SessionSummary,
    constrain_summary,
    estimate_summary_tokens,
    prepend_summary_message,
    strip_summary_message,
    summary_prompt_budget,
)

if TYPE_CHECKING:
    from .types import Session, SessionEvent, SessionTranscriptHead, SessionTurn

# NOTE: `.types` imports this module at module level (RemoteHistoryStore is used
# as a dataclass field default there), so importing `.types` symbols at module
# level here would create a circular import. Type-only references stay in the
# TYPE_CHECKING block above; symbols needed at runtime (e.g. RemoteSession,
# SessionTranscriptHead) are imported locally inside the methods/functions below
# instead of being hoisted to module level.


def _events_to_raw_messages(events: "list[SessionEvent]") -> "list[Any]":
    result = []
    for event in events:
        message = (event.payload or {}).get("message")
        if message:
            result.append(message)
    return result


def _load_session_summary(meta_json: "dict[str, Any] | None") -> "SessionSummary | None":
    if not isinstance(meta_json, dict):
        return None
    return SessionSummary.from_dict(meta_json.get("summary") if isinstance(meta_json.get("summary"), dict) else None)


def _unsummarized_trimmed_messages(
    trimmed_messages: "list[ModelMessage]",
    *,
    existing_summary: "SessionSummary | None",
    covered_until_seq: int,
) -> "list[ModelMessage]":
    if not trimmed_messages:
        return []
    already_covered_until = existing_summary.covered_until_seq if existing_summary is not None else 0
    unsummarized_count = covered_until_seq - already_covered_until
    if unsummarized_count <= 0:
        return []
    if unsummarized_count >= len(trimmed_messages):
        return list(trimmed_messages)
    return list(trimmed_messages[-unsummarized_count:])


class RemoteHistoryStore:
    def __init__(self, store: TranscriptStore) -> None:
        self._store = store

    async def load(self, session: "Session") -> "list[ModelMessage]":
        from .types import RemoteSession

        if isinstance(session, RemoteSession) and session.loaded_head_seq > 0 and session.loaded_messages:
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
            if isinstance(session, RemoteSession):
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
        if isinstance(session, RemoteSession):
            session.history_token_budget = effective_budget
            session.loaded_head_seq = transcript.loaded_until_seq or transcript.head.head_seq or transcript.head.snapshot_seq
            session.loaded_messages = list(window.recent_messages)
            session.loaded_summary = summary
        return prepend_summary_message(window.recent_messages, prompt_summary)

    async def persist(self, session: "Session", turn: "SessionTurn") -> None:
        from .types import RemoteSession, SessionTranscriptHead, SessionTranscriptWrite

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
        if request_id is not None and isinstance(session, RemoteSession):
            session.pending_request_id = None
        head_seq = prior_head_seq + len(events)
        await self._store.save(
            SessionTranscriptWrite(
                session_id=session.session_id,
                events=events,
                head=SessionTranscriptHead(
                    session_kind=type(session).__name__,
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
        if isinstance(session, RemoteSession):
            session.loaded_head_seq = head_seq
            session.loaded_messages = list(window.recent_messages)
            session.history_token_budget = history_token_budget
            session.loaded_summary = summary


def _build_session_events(
    messages: "list[ModelMessage]",
    *,
    start_seq: int,
    session_id: str,
    turn_key: str,
    call_id: "str | None",
    seed_display_entries: "list[dict[str, Any]] | None" = None,
    request_id: "str | None" = None,
) -> "list[SessionEvent]":
    from .types import SessionEvent

    events: "list[SessionEvent]" = []
    dumped_messages = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    known_tool_names: "dict[str, str]" = {}
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
        payload: "dict[str, Any]" = {
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
    entries: "list[dict[str, Any]]",
    *,
    known_tool_names: "dict[str, str]",
    seed_display_entries: "list[dict[str, Any]] | None",
) -> "list[dict[str, Any]]":
    from .types import normalize_display_entry, normalize_display_entries

    display_entries: "list[dict[str, Any]]" = []
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


def _primary_display_text(display_entries: "list[dict[str, Any]]") -> str:
    for item in display_entries:
        text = item.get("text")
        if isinstance(text, str) and text.strip():
            return text
    return ""


def _estimate_dumped_tokens(payload: Any) -> int:
    text = json.dumps(payload, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


def _estimate_model_messages(messages: "list[ModelMessage]") -> int:
    if not messages:
        return 0
    return _estimate_dumped_tokens(ModelMessagesTypeAdapter.dump_python(messages, mode="json"))


def _resolved_history_token_budget(session: "Session", head: "SessionTranscriptHead") -> int:
    configured_budget = getattr(session, "history_token_budget", 24000)
    if head.is_empty:
        return configured_budget
    stored_budget = int(head.history_token_budget or configured_budget)
    return min(configured_budget, stored_budget)
