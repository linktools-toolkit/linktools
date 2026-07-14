#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session window policies: context-window trimming as a pluggable
policy, separate from SessionStore (which only stores/retrieves).

- NoopWindowPolicy: pass every message through (default).
- RecentWindowPolicy: keep the most recent N messages.
The Runtime applies the selected policy before handing messages to the model."""

from typing import TYPE_CHECKING, Protocol, Sequence, runtime_checkable

if TYPE_CHECKING:
    from ..model.policy import ModelPolicy
    from ..session.models import SessionMessage


@runtime_checkable
class SessionWindowPolicy(Protocol):
    async def select_messages(
        self,
        messages: "Sequence[SessionMessage]",
        model_policy: "ModelPolicy",
    ) -> "Sequence[SessionMessage]": ...


class NoopWindowPolicy:
    """Returns all messages unchanged."""

    async def select_messages(self, messages, model_policy):  # type: ignore[no-untyped-def]
        return list(messages)


class RecentWindowPolicy:
    """Keep only the most recent ``max_messages`` messages (defaults to 20)."""

    def __init__(self, *, max_messages: int = 20) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self._max = max_messages

    async def select_messages(self, messages, model_policy):  # type: ignore[no-untyped-def]
        if len(messages) <= self._max:
            return list(messages)
        return list(messages[-self._max :])
