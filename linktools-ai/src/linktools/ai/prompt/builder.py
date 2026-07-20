#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PromptBuilder: owns model-prompt template composition. Folds session
history, memory, knowledge, and capability-resolved prompt sections around the
user prompt into the single string sent to the model. Plan §4.2: the prompt
domain owns PromptSpec + PromptBuilder + template composition, and does not
read the filesystem -- the runner hands in already-fetched sections (memory,
knowledge, capability catalog) and the builder only composes.

Two distinct values are composed and the distinction is load-bearing:

* ``user_prompt`` is the caller's ORIGINAL input (what gets persisted as the
  USER session message); it never carries runtime context.
* the composed model prompt returned here is history + memory + knowledge +
  capability sections + user -- never persisted verbatim, so internal runtime
  context cannot leak into session history.

The methods are pure string transforms over already-fetched inputs; async
policy execution (memory recall, knowledge retrieval, capability resolution)
stays with the caller."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ..session.models import SessionMessage


class PromptBuilder:
    """Model-prompt template composition. Stateless; every method is a pure
    string transform."""

    @staticmethod
    def format_session_history(messages: "Sequence[SessionMessage]") -> str:
        """Render prior session messages into the MODEL prompt with role
        prefixes, so an assistant turn is not disguised as user content. This
        is injected into the model prompt only -- the persisted USER session
        message is always the caller's original prompt, never this rendering."""
        lines: "list[str]" = []
        for message in messages:
            role = message.role.value.upper()
            content = message.content
            if not isinstance(content, str):
                content = repr(content)
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def build_base_prompt(
        *,
        user_prompt: str,
        prior_messages: "Sequence[SessionMessage]",
        memory_section: str = "",
        knowledge_section: str = "",
    ) -> str:
        """Compose the base model prompt: session history + user prompt, with
        optional memory and knowledge sections prepended.

        Final top-to-bottom order when both sections are non-empty:
        ``knowledge`` on top, then ``memory``, then session history, then the
        user prompt. Empty sections are skipped (never add a blank line)."""
        history_text = PromptBuilder.format_session_history(prior_messages)
        prompt = (
            f"{history_text}\n{user_prompt}" if history_text else user_prompt
        )
        if memory_section:
            prompt = f"{memory_section}\n{prompt}"
        if knowledge_section:
            prompt = f"{knowledge_section}\n{prompt}"
        return prompt

    @staticmethod
    def combine(
        *,
        base_prompt: "str | None",
        capability_sections: "Mapping[str, str]",
        static_sections: "Mapping[str, str] | None" = None,
        resuming: bool,
    ) -> "str | None":
        """Merge the spec-declared static sections + the capability-resolved
        sections with the base prompt.

        ``static_sections`` are the agent's declared ``PromptSpec.sections``
        (stable, spec-authored); ``capability_sections`` are the dynamically
        resolved catalog sections (skills/extensions/etc.). Both render the
        same way -- joined into one catalog prompt prepended to the base.
        Static sections come first (agent identity), then capability sections
        (dynamic catalog); a colliding key is won by the capability section.

        On the resume path (``resuming=True``) the prompt is baked into the
        checkpointed ``message_history`` -- return ``None`` so the runner does
        not re-feed a prompt alongside the history."""
        if resuming:
            return None
        merged: "dict[str, str]" = {}
        if static_sections:
            merged.update(static_sections)
        if capability_sections:
            merged.update(capability_sections)
        catalog_prompt = "\n\n".join(merged.values()) if merged else ""
        if catalog_prompt:
            return f"{catalog_prompt}\n\n{base_prompt}"
        return base_prompt
