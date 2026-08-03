#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WorkspaceScreen integration tests via Textual's Pilot.

Drives the persistent three-pane workspace through a FakeRuntimeClient. These
tests cover submit→stream→render, markup safety, cancel-via-runtime, one-run-
at-a-time, slash commands, and the sidebar/context refresh wiring. They are the
canonical smoke test that the TUI actually composes and renders (an import-only
check cannot catch layout/CSS/compose errors)."""

import asyncio

import pytest

from linktools.ai.cli.client import FakeRuntimeClient
from linktools.ai.cli.tui.app import LinktoolsAIApp
from linktools.ai.cli.tui.widgets import (
    Composer,
    ConversationView,
    ContextPanel,
    SessionSelected,
    Sidebar,
    StatusBar,
)


def _client(**kwargs) -> FakeRuntimeClient:
    return FakeRuntimeClient(**kwargs)


async def _wait_for(pred, *, timeout: float = 2.0) -> bool:
    """Poll pred() until truthy or timeout; returns whether it succeeded."""
    elapsed = 0.0
    step = 0.02
    while elapsed < timeout:
        if pred():
            return True
        await asyncio.sleep(step)
        elapsed += step
    return bool(pred())


async def _set_and_submit(composer: Composer, text: str, pilot) -> None:
    composer.query_one("#composer-input").text = text
    composer.action_submit()
    await pilot.pause()


@pytest.mark.asyncio
async def test_submit_streams_and_renders_assistant_text() -> None:
    client = _client(stream_events=[{"type": "text", "text": "Hello"}])
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "hi", pilot)
        await _wait_for(
            lambda: any(t.kind == "assistant" for t in ws.conversation.turns)
        )
        conv = ws.conversation
        assert any(t.kind == "user" and t.text == "hi" for t in conv.turns)
        assert any(t.kind == "assistant" and "Hello" in t.text for t in conv.turns)
        assert client.run_requests and client.run_requests[0].prompt == "hi"


@pytest.mark.asyncio
async def test_untrusted_text_is_markup_safe() -> None:
    # A stray [/b] in model output must not raise MarkupError on render.
    client = _client(stream_events=[{"type": "text", "text": "evil [/b] [red] done"}])
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "go", pilot)
        await _wait_for(
            lambda: any(t.kind == "assistant" for t in ws.conversation.turns)
        )
        ws.query_one(ConversationView).refresh_from()
        assert any(
            "[/b]" in t.text for t in ws.conversation.turns if t.kind == "assistant"
        )


@pytest.mark.asyncio
async def test_one_run_at_a_time() -> None:
    # stream_events that never end keeps the first worker busy; a second submit
    # while it is active is refused.
    blocker = asyncio.Event()

    class _BlockingClient(FakeRuntimeClient):
        async def run_stream(self, request):
            self.run_requests.append(request)
            self.last_run_id = request.run_id
            yield {"type": "text", "text": "..."}
            await blocker.wait()

    client = _BlockingClient(stream_events=[])
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "first", pilot)
        await _wait_for(lambda: ws._active_worker is not None)
        await _set_and_submit(ws.query_one(Composer), "second", pilot)
        assert len(client.run_requests) == 1
        assert client.run_requests[0].prompt == "first"
        blocker.set()
        await _wait_for(lambda: ws._active_worker is None)


@pytest.mark.asyncio
async def test_cancel_signals_runtime() -> None:
    blocker = asyncio.Event()

    class _BlockingClient(FakeRuntimeClient):
        async def run_stream(self, request):
            self.run_requests.append(request)
            self.last_run_id = request.run_id
            yield {"type": "text", "text": "..."}
            await blocker.wait()

    client = _BlockingClient(stream_events=[])
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "run", pilot)
        await _wait_for(lambda: ws._active_worker is not None)
        ws.action_cancel_run()
        await _wait_for(lambda: bool(client.cancel_calls))
        assert client.cancel_calls
        blocker.set()


@pytest.mark.asyncio
async def test_slash_clear_clears_conversation() -> None:
    client = _client()
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        ws.conversation.add_user("x")
        ws.conversation.add_user("y")
        assert len(ws.conversation.turns) == 2
        await _set_and_submit(ws.query_one(Composer), "/clear", pilot)
        await pilot.pause()
        assert ws.conversation.turns == ()


@pytest.mark.asyncio
async def test_slash_new_switches_session() -> None:
    client = _client()
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        old = ws.session_id
        await _set_and_submit(ws.query_one(Composer), "/new fresh-session", pilot)
        await pilot.pause()
        assert ws.session_id == "fresh-session"
        assert ws.session_id != old
        assert ws.query_one(Sidebar)._active_session == "fresh-session"


@pytest.mark.asyncio
async def test_slash_help_is_not_a_prompt() -> None:
    client = _client()
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "/help", pilot)
        await pilot.pause()
        assert client.run_requests == []
        assert any(t.kind == "status" for t in ws.conversation.turns)


@pytest.mark.asyncio
async def test_status_bar_reflects_run_state() -> None:
    client = _client(stream_events=[{"type": "text", "text": "ok"}])
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "go", pilot)
        await _wait_for(
            lambda: any(t.kind == "assistant" for t in ws.conversation.turns)
        )
        await pilot.pause()
        bar = ws.query_one(StatusBar)
        assert ws.session_id in bar.left_text


@pytest.mark.asyncio
async def test_workspace_layout_has_three_panes() -> None:
    client = _client()
    app = LinktoolsAIApp(client=client)
    async with app.run_test():
        ws = app.workspace
        assert ws.query_one(Sidebar) is not None
        assert ws.query_one(ConversationView) is not None
        assert ws.query_one(ContextPanel) is not None


@pytest.mark.asyncio
async def test_approval_modal_opens_on_pause() -> None:
    paused_event = {
        "type": "paused",
        "run_id": "run-1",
        "approval_id": "ap-1",
    }
    client = _client(stream_events=[paused_event])
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "go", pilot)
        await _wait_for(lambda: len(app.screen_stack) > 1)
        # The approval modal is now on top of the workspace.
        from linktools.ai.cli.tui.modals import ApprovalModal

        assert isinstance(app.screen, ApprovalModal)


@pytest.mark.asyncio
async def test_session_selection_loads_history() -> None:
    history = (
        (
            {
                "kind": "request",
                "parts": [{"type": "text", "content": "prior question"}],
            },
            {
                "kind": "response",
                "parts": [{"type": "text", "content": "prior answer"}],
            },
        ),
    )
    client = _client(session_messages=history)
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        ws.post_message(SessionSelected("loaded-session"))
        await pilot.pause()
        await _wait_for(
            lambda: any(t.kind == "assistant" for t in ws.conversation.turns)
        )
        assert ws.session_id == "loaded-session"
        assert any(
            t.kind == "assistant" and "prior answer" in t.text
            for t in ws.conversation.turns
        )


@pytest.mark.asyncio
async def test_f1_opens_help_modal() -> None:
    client = _client()
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        app.workspace.query_one("#composer-input").blur()
        await pilot.pause()
        await pilot.press("f1")
        await pilot.pause()
        from linktools.ai.cli.tui.modals import HelpModal

        assert isinstance(app.screen, HelpModal)
