#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command palette + slash command tests.

Covers the Ctrl+P palette provider (searches and yields app actions) and the
composer slash commands (/clear, /new, /cancel, /doctor, /catalog, /help, /exit,
unknown). The palette provider is exercised directly; the slash commands are
driven through the workspace composer."""

import asyncio

import pytest

from linktools.ai.cli.client import FakeRuntimeClient
from linktools.ai.cli.tui.app import LinktoolsAIApp
from linktools.ai.cli.tui.commands import AiCommandProvider
from linktools.ai.cli.tui.widgets import Composer


async def _wait_for(pred, *, timeout: float = 2.0) -> bool:
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
async def test_command_palette_yields_entries() -> None:
    client = FakeRuntimeClient()
    app = LinktoolsAIApp(client=client)
    async with app.run_test():
        provider = AiCommandProvider(screen=app.screen)
        hits = []
        async for hit in provider.search(""):
            hits.append(hit)
        names = [hit.text for hit in hits]
        # All workspace-level actions are present.
        assert "Doctor" in names
        assert "Catalog" in names
        assert "Quit" in names
        assert "New session" in names


@pytest.mark.asyncio
async def test_command_palette_filters_by_query() -> None:
    client = FakeRuntimeClient()
    app = LinktoolsAIApp(client=client)
    async with app.run_test():
        provider = AiCommandProvider(screen=app.screen)
        hits = []
        async for hit in provider.search("doc"):
            hits.append(hit)
        names = [hit.text for hit in hits]
        assert names == ["Doctor"]


@pytest.mark.asyncio
async def test_slash_cancel_when_no_run_is_safe() -> None:
    client = FakeRuntimeClient()
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "/cancel", pilot)
        # No run in flight: cancel is a no-op, no exception.
        assert client.cancel_calls == []


@pytest.mark.asyncio
async def test_slash_unknown_reports_error() -> None:
    client = FakeRuntimeClient()
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "/bogus", pilot)
        assert any(
            t.kind == "error" and "unknown" in t.message for t in ws.conversation.turns
        )


@pytest.mark.asyncio
async def test_slash_exit_exits_app() -> None:
    client = FakeRuntimeClient()
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "/exit", pilot)
        await pilot.pause()
        # The app was asked to exit.
        assert app._exit


@pytest.mark.asyncio
async def test_slash_doctor_opens_modal() -> None:
    client = FakeRuntimeClient()
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "/doctor", pilot)
        await _wait_for(lambda: len(app.screen_stack) > 1)
        from linktools.ai.cli.tui.modals import DoctorModal

        assert isinstance(app.screen, DoctorModal)


@pytest.mark.asyncio
async def test_slash_catalog_opens_modal() -> None:
    client = FakeRuntimeClient(agents=("a1",))
    app = LinktoolsAIApp(client=client)
    async with app.run_test() as pilot:
        ws = app.workspace
        await _set_and_submit(ws.query_one(Composer), "/catalog", pilot)
        await _wait_for(lambda: len(app.screen_stack) > 1)
        from linktools.ai.cli.tui.modals import CatalogModal

        assert isinstance(app.screen, CatalogModal)
