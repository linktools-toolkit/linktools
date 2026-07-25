#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SessionRecorder: message-format conversion for a completed or paused agent
turn. Owns turning a raw user prompt + model output into the
``NewSessionMessage`` shape ``SessionStore.append_messages`` accepts -- it does
NOT own Run state (no RunStore/CheckpointStore access, no transitions) and
never touches SessionStore itself; the actual cross-store write happens
inside the RunCommitCoordinator this module's output is handed to."""

from .models import MessageRole, NewSessionMessage


class SessionRecorder:
    """Stateless message-format converter. A plain class (not a set of
    module functions) so per-tenant/per-format configuration can later be
    carried as instance state without changing every call site's signature."""

    def completed_messages(
        self,
        *,
        user_prompt: str,
        output: object,
        run_id: str,
    ) -> "tuple[NewSessionMessage, ...]":
        """Build the USER + ASSISTANT message pair for one completed turn.
        The USER message is omitted when ``user_prompt`` is empty (a resume
        continuation has no new user turn to record)."""
        messages: "list[NewSessionMessage]" = [
            NewSessionMessage(
                role=MessageRole.ASSISTANT,
                content=str(output),
                run_id=run_id,
            ),
        ]
        if user_prompt:
            messages.insert(
                0,
                NewSessionMessage(
                    role=MessageRole.USER,
                    content=user_prompt,
                    run_id=run_id,
                ),
            )
        return tuple(messages)

    def paused_messages(self, *, run_id: str) -> "tuple[NewSessionMessage, ...]":
        """A pause never closes out a session turn -- no message is appended
        until the run later completes or fails. ``run_id`` is accepted (over
        returning a bare constant) so a future policy could record a
        turn-in-progress marker without changing the call site's shape."""
        del run_id
        return ()


__all__: "list[str]" = ["SessionRecorder"]
