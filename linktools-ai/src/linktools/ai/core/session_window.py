from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Protocol

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelRequest, SystemPromptPart

from .model_runtime import _model_message_to_dicts, _trim_messages

DEFAULT_SUMMARY_MAX_TOKENS = 256
SUMMARY_PROMPT_BUDGET_RATIO_NUMERATOR = 3
SUMMARY_PROMPT_BUDGET_RATIO_DENOMINATOR = 4
SUMMARY_MAX_FACTS = 12
SUMMARY_MAX_OPEN_THREADS = 8
SUMMARY_MAX_ITEM_CHARS = 240
SUMMARY_MAX_TEXT_CHARS = 1200


def _estimate_window_tokens(messages: list[ModelMessage]) -> int:
    if not messages:
        return 0
    payload = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    return max(1, len(json.dumps(payload, ensure_ascii=False, default=str)) // 4)


def _message_content(message: ModelMessage) -> str | None:
    for entry in _model_message_to_dicts(message):
        content = str(entry.get("content") or "").strip()
        if content:
            return content
    return None


@dataclass(frozen=True, slots=True)
class SessionSummary:
    summary_text: str = ""
    covered_until_seq: int = 0
    facts: list[str] = field(default_factory=list)
    open_threads: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        normalized = constrain_summary(self)
        return {
            "summary_text": _summary_text(normalized.summary_text, normalized.facts, normalized.open_threads)
            if normalized is not None
            else "",
            "covered_until_seq": int(normalized.covered_until_seq if normalized is not None else self.covered_until_seq or 0),
            "facts": list(normalized.facts if normalized is not None else self.facts),
            "open_threads": list(normalized.open_threads if normalized is not None else self.open_threads),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "SessionSummary | None":
        if not isinstance(payload, dict):
            return None
        facts = [str(item) for item in (payload.get("facts") or []) if str(item).strip()]
        open_threads = [str(item) for item in (payload.get("open_threads") or []) if str(item).strip()]
        summary_text = _summary_text(str(payload.get("summary_text") or ""), facts, open_threads)
        covered_until_seq = int(payload.get("covered_until_seq") or 0)
        if covered_until_seq <= 0 and not summary_text and not facts and not open_threads:
            return None
        return constrain_summary(
            cls(
                summary_text=summary_text,
                covered_until_seq=covered_until_seq,
                facts=facts,
                open_threads=open_threads,
            )
        )


@dataclass(frozen=True, slots=True)
class SessionWindow:
    summary: SessionSummary | None
    recent_messages: list[ModelMessage]
    trimmed_messages: list[ModelMessage]
    snapshot_boundary_seq: int


class SessionWindowPolicy(Protocol):
    def build(
        self,
        messages: list[ModelMessage],
        *,
        budget_tokens: int,
        head_seq: int,
        summary: SessionSummary | None = None,
    ) -> SessionWindow:
        ...


class SessionSummaryPolicy(Protocol):
    def summarize(
        self,
        trimmed_messages: list[ModelMessage],
        *,
        existing_summary: SessionSummary | None,
        covered_until_seq: int,
    ) -> SessionSummary | None:
        ...


class RecentWindowPolicy:
    def build(
        self,
        messages: list[ModelMessage],
        *,
        budget_tokens: int,
        head_seq: int,
        summary: SessionSummary | None = None,
    ) -> SessionWindow:
        trimmed = _trim_messages(messages)
        pretrimmed_count = max(0, len(messages) - len(trimmed))
        trimmed_messages = list(messages[:pretrimmed_count])
        summary = constrain_summary(summary)
        prompt_summary = constrain_summary(summary, max_tokens=summary_prompt_budget(budget_tokens))
        if not trimmed and not trimmed_messages:
            return SessionWindow(summary=summary, recent_messages=[], trimmed_messages=[], snapshot_boundary_seq=head_seq)

        recent_messages = list(trimmed)
        recent_budget = max(0, budget_tokens - estimate_summary_tokens(prompt_summary))
        total = _estimate_window_tokens(recent_messages)

        while recent_messages and total > recent_budget:
            trimmed_messages.append(recent_messages.pop(0))
            total = _estimate_window_tokens(recent_messages)

        return SessionWindow(
            summary=summary,
            recent_messages=recent_messages,
            trimmed_messages=trimmed_messages,
            snapshot_boundary_seq=max(0, head_seq - len(recent_messages)),
        )


class RollingSummaryPolicy:
    def summarize(
        self,
        trimmed_messages: list[ModelMessage],
        *,
        existing_summary: SessionSummary | None,
        covered_until_seq: int,
    ) -> SessionSummary | None:
        facts = list(existing_summary.facts) if existing_summary else []
        open_threads = list(existing_summary.open_threads) if existing_summary else []
        legacy_summary_text = str(existing_summary.summary_text).strip() if existing_summary else ""
        new_items: list[str] = []
        for message in trimmed_messages:
            content = _message_content(message)
            if content:
                facts.append(content)
                new_items.append(content)
        facts = _normalize_summary_items(facts, max_items=SUMMARY_MAX_FACTS)
        open_threads = _normalize_summary_items(open_threads, max_items=SUMMARY_MAX_OPEN_THREADS)
        if legacy_summary_text and not (existing_summary and existing_summary.facts):
            appended = "\n".join(item for item in new_items if item)
            summary_text = "\n".join(part for part in (legacy_summary_text, appended) if part).strip()
        else:
            summary_text = _summary_text("", facts, open_threads)
        if covered_until_seq <= 0 and not summary_text and not facts and not open_threads:
            return None
        return constrain_summary(
            SessionSummary(
                summary_text=summary_text,
                covered_until_seq=covered_until_seq,
                facts=facts,
                open_threads=open_threads,
            )
        )


class NoopSummaryPolicy:
    def summarize(
        self,
        trimmed_messages: list[ModelMessage],
        *,
        existing_summary: SessionSummary | None,
        covered_until_seq: int,
    ) -> SessionSummary | None:
        del trimmed_messages, existing_summary, covered_until_seq
        return None


def build_summary_message(summary: SessionSummary | None) -> ModelMessage | None:
    if summary is None:
        return None
    text = render_summary_block(summary)
    if not text:
        return None
    return ModelRequest(parts=[SystemPromptPart(text)])


def prepend_summary_message(
    messages: list[ModelMessage],
    summary: SessionSummary | None,
) -> list[ModelMessage]:
    summary_message = build_summary_message(summary)
    if summary_message is None:
        return list(messages)
    return [summary_message, *messages]


def strip_summary_message(
    messages: list[ModelMessage],
    summary: SessionSummary | None,
) -> list[ModelMessage]:
    if not messages:
        return []
    rendered = render_summary_block(summary)
    if not rendered:
        return list(messages)
    first = messages[0]
    if isinstance(first, ModelRequest):
        for part in first.parts:
            if isinstance(part, SystemPromptPart) and str(part.content or "").strip() == rendered:
                return list(messages[1:])
    return list(messages)


def render_summary_block(summary: SessionSummary | None) -> str:
    if summary is None:
        return ""
    summary_text = _summary_text(summary.summary_text, summary.facts, summary.open_threads)
    if not summary_text and not summary.open_threads:
        return ""
    lines = ["Conversation summary:"]
    if summary_text:
        lines.append(summary_text)
    if summary.open_threads:
        lines.append("Open threads:")
        lines.extend(f"- {item}" for item in summary.open_threads)
    return "\n".join(line for line in lines if line).strip()


def _summary_text(summary_text: str, facts: list[str], open_threads: list[str]) -> str:
    normalized = str(summary_text or "").strip()
    if normalized:
        return normalized
    lines = [str(item).strip() for item in facts if str(item).strip()]
    del open_threads
    return "\n".join(lines).strip()


def summary_prompt_budget(budget_tokens: int) -> int:
    if budget_tokens <= 0:
        return 0
    proportional_budget = max(1, (budget_tokens * SUMMARY_PROMPT_BUDGET_RATIO_NUMERATOR) // SUMMARY_PROMPT_BUDGET_RATIO_DENOMINATOR)
    return min(DEFAULT_SUMMARY_MAX_TOKENS, proportional_budget)


def estimate_summary_tokens(summary: SessionSummary | None) -> int:
    message = build_summary_message(summary)
    if message is None:
        return 0
    return _estimate_window_tokens([message])


def constrain_summary(
    summary: SessionSummary | None,
    *,
    max_tokens: int = DEFAULT_SUMMARY_MAX_TOKENS,
) -> SessionSummary | None:
    if summary is None:
        return None

    covered_until_seq = max(0, int(summary.covered_until_seq or 0))
    facts = _normalize_summary_items(summary.facts, max_items=SUMMARY_MAX_FACTS)
    open_threads = _normalize_summary_items(summary.open_threads, max_items=SUMMARY_MAX_OPEN_THREADS)
    summary_text = _truncate_summary_text(
        _summary_text(summary.summary_text, facts, open_threads),
        max_chars=SUMMARY_MAX_TEXT_CHARS,
    )

    while estimate_summary_tokens(
        SessionSummary(
            summary_text=summary_text,
            covered_until_seq=covered_until_seq,
            facts=facts,
            open_threads=open_threads,
        )
    ) > max_tokens:
        if open_threads:
            open_threads = open_threads[1:]
            continue
        next_text = _shrink_summary_text(summary_text)
        if next_text == summary_text:
            break
        summary_text = next_text

    if covered_until_seq <= 0 and not summary_text and not facts and not open_threads:
        return None
    constrained = SessionSummary(
        summary_text=summary_text,
        covered_until_seq=covered_until_seq,
        facts=facts,
        open_threads=open_threads,
    )
    if estimate_summary_tokens(constrained) > max_tokens:
        return None
    return constrained


def _normalize_summary_items(items: list[str], *, max_items: int) -> list[str]:
    normalized = []
    for item in items:
        text = _truncate_summary_text(str(item or "").strip(), max_chars=SUMMARY_MAX_ITEM_CHARS)
        if text:
            normalized.append(text)
    if len(normalized) <= max_items:
        return normalized
    return normalized[-max_items:]


def _truncate_summary_text(text: str, *, max_chars: int) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 3:
        return normalized[:max_chars]
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _shrink_summary_text(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    step = max(32, len(normalized) // 4)
    candidate = normalized[: max(1, len(normalized) - step)].rstrip()
    if not candidate:
        candidate = normalized[:1]
    return _truncate_summary_text(candidate, max_chars=SUMMARY_MAX_TEXT_CHARS)
