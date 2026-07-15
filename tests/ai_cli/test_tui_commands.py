#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command palette + slash command tests (spec §20/§32).

Slash commands (/help, /new, /clear, ...) are dispatched from the composer and
must not be sent to the agent. The command palette (Ctrl+P) provider is
registered on the App."""

import unittest

from textual.widgets import Input, RichLog

from linktools.ai_cli.client import FakeRuntimeClient
from linktools.ai_cli.tui.app import LinktoolsAIApp
from linktools.ai_cli.tui.commands import AiCommandProvider


async def _wait_until(pilot, cond, *, tries: int = 50) -> bool:
    for _ in range(tries):
        if cond():
            return True
        await pilot.pause()
    return cond()


def _conv_recorder(pilot) -> list:
    conv = pilot.app.screen.query_one("#conversation", RichLog)
    recorded: "list[str]" = []
    conv.write = recorded.append  # type: ignore[method-assign]
    return recorded


class TestSlashCommands(unittest.IsolatedAsyncioTestCase):
    async def test_help_slash_command(self):
        fake = FakeRuntimeClient()
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            recorded = _conv_recorder(pilot)
            pilot.app.screen.query_one(Input).value = "/help"
            await pilot.press("enter")
            await _wait_until(pilot, lambda: any("slash" in str(x) for x in recorded))
        self.assertTrue(any("slash" in str(x) for x in recorded))

    async def test_new_session_sets_session_id(self):
        fake = FakeRuntimeClient()
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            recorded = _conv_recorder(pilot)
            pilot.app.screen.query_one(Input).value = "/new test-session"
            await pilot.press("enter")
            await _wait_until(
                pilot, lambda: any("test-session" in str(x) for x in recorded)
            )
            self.assertEqual(pilot.app.screen.session_id, "test-session")

    async def test_unknown_slash_does_not_send_to_agent(self):
        fake = FakeRuntimeClient(stream_events=[{"type": "text", "text": "hello"}])
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            pilot.app.screen.query_one(Input).value = "/bogus"
            await pilot.press("enter")
            await pilot.pause()
        self.assertEqual(fake.run_requests, [])


class TestCommandPaletteRegistration(unittest.TestCase):
    def test_provider_registered(self):
        self.assertIn(AiCommandProvider, LinktoolsAIApp.COMMANDS)


if __name__ == "__main__":
    unittest.main()
