#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PromptBuilder (contract): pure model-prompt template composition. Owned by
the prompt domain per plan §4.2; the runner hands in already-fetched sections
and the builder only composes."""

from datetime import datetime, timezone

from linktools.ai.prompt.builder import PromptBuilder
from linktools.ai.session.models import MessageRole, SessionMessage

_NOW = datetime.now(timezone.utc)


def _msg(role: MessageRole, content, mid: str = "m"):
    return SessionMessage(
        id=mid,
        session_id="s",
        sequence=0,
        role=role,
        content=content,
        run_id=None,
        created_at=_NOW,
    )


def test_format_session_history_uses_role_prefixes():
    out = PromptBuilder.format_session_history(
        [_msg(MessageRole.ASSISTANT, "hi"), _msg(MessageRole.USER, "ok")]
    )
    # An assistant turn must NOT be disguised as user content.
    assert out == "ASSISTANT: hi\nUSER: ok"


def test_format_session_history_reprs_non_string_content():
    out = PromptBuilder.format_session_history(
        [_msg(MessageRole.USER, {"k": "v"})]
    )
    assert "USER:" in out and "k" in out and "v" in out


def test_build_base_prompt_user_only():
    assert (
        PromptBuilder.build_base_prompt(
            user_prompt="hello", prior_messages=[]
        )
        == "hello"
    )


def test_build_base_prompt_prepends_history():
    out = PromptBuilder.build_base_prompt(
        user_prompt="hello",
        prior_messages=[_msg(MessageRole.USER, "prior")],
    )
    assert out == "USER: prior\nhello"


def test_build_base_prompt_memory_and_knowledge_order():
    # Final order top-to-bottom: knowledge, memory, history, user.
    out = PromptBuilder.build_base_prompt(
        user_prompt="u",
        prior_messages=[_msg(MessageRole.USER, "h")],
        memory_section="## Memory",
        knowledge_section="## Knowledge",
    )
    assert out == "## Knowledge\n## Memory\nUSER: h\nu"


def test_build_base_prompt_empty_sections_skipped():
    # Empty memory/knowledge must not add blank lines.
    out = PromptBuilder.build_base_prompt(
        user_prompt="u",
        prior_messages=[],
        memory_section="",
        knowledge_section="",
    )
    assert out == "u"


def test_combine_prepends_capability_sections():
    out = PromptBuilder.combine(
        base_prompt="base",
        capability_sections={"skills": "## Skills", "mcp": "## MCP"},
        resuming=False,
    )
    assert out == "## Skills\n\n## MCP\n\nbase"


def test_combine_no_sections_returns_base():
    assert (
        PromptBuilder.combine(
            base_prompt="base", capability_sections={}, resuming=False
        )
        == "base"
    )


def test_combine_returns_none_on_resume():
    # On resume the prompt is baked into checkpointed message_history; the
    # runner must not re-feed a prompt alongside it.
    assert (
        PromptBuilder.combine(
            base_prompt="base",
            capability_sections={"skills": "## Skills"},
            resuming=True,
        )
        is None
    )


def test_combine_prepends_static_sections():
    # PromptSpec.sections (the agent's declared static sections) render the
    # same way as capability sections, static first.
    out = PromptBuilder.combine(
        base_prompt="base",
        capability_sections={"skills": "## Skills"},
        static_sections={"persona": "You are a grader."},
        resuming=False,
    )
    assert out == "You are a grader.\n\n## Skills\n\nbase"


def test_combine_static_only():
    assert (
        PromptBuilder.combine(
            base_prompt="base",
            capability_sections={},
            static_sections={"persona": "You are a grader."},
            resuming=False,
        )
        == "You are a grader.\n\nbase"
    )


def test_combine_capability_key_overrides_static_on_collision():
    # A colliding key is won by the capability section (dynamic catalog
    # overrides the static declaration).
    out = PromptBuilder.combine(
        base_prompt="base",
        capability_sections={"persona": "dynamic"},
        static_sections={"persona": "static"},
        resuming=False,
    )
    assert out == "dynamic\n\nbase"
