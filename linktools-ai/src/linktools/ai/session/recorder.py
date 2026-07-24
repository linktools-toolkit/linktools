#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SessionRecorder: message-format conversion for a completed agent turn
(spec section 12.6). Owns turning a raw user prompt + model output into the
``NewSessionMessage`` shape ``SessionStore.append_messages`` accepts -- it does
NOT own Run state (no RunStore/CheckpointStore access, no transitions); the
actual cross-store write happens inside the RunCommitCoordinator this
module's output is handed to, exactly as before this extraction."""

from .models import MessageRole, NewSessionMessage


class SessionRecorder:
    """Stateless message-format converter. A plain class (not a set of
    module functions) so a future increment can carry per-tenant/per-format
    configuration without changing every call site's signature."""

    def build_turn_messages(
        self,
        *,
        user_prompt: str,
        output: object,
        run_id: str,
    ) -> "tuple[NewSessionMessage, ...]":
        """Build the USER + ASSISTANT message pair for one completed turn.
        The USER message is omitted when ``user_prompt`` is empty (a resume
        continuation has no new user turn to record) -- matching the
        pre-extraction behavior exactly."""
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


__all__: "list[str]" = ["SessionRecorder"]
