#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resources/Runs/Doctor screen + navigation tests (spec §17.2/§17.3/§17.4/§32).

Each screen fetches its data through ``RuntimeClient`` (no registry/storage
access from the UI). Ctrl+R/O/D push the screens; Esc pops back to chat."""

import unittest

from linktools.ai_cli.client import DoctorReport, FakeRuntimeClient
from linktools.ai_cli.tui.app import LinktoolsAIApp
from linktools.ai_cli.tui.screens.chat import ChatScreen
from linktools.ai_cli.tui.screens.doctor import DoctorScreen
from linktools.ai_cli.tui.screens.resources import ResourcesScreen
from linktools.ai_cli.tui.screens.runs import RunsScreen


async def _wait_until(pilot, cond, *, tries: int = 50) -> bool:
    for _ in range(tries):
        if cond():
            return True
        await pilot.pause()
    return cond()


def _spy(fake, method_name: str, record: list):
    """Wrap ``fake.<method_name>`` to append its name to ``record`` on call."""
    orig = getattr(fake, method_name)

    async def wrapped(*args, **kwargs):
        record.append(method_name)
        return await orig(*args, **kwargs)

    setattr(fake, method_name, wrapped)


class TestTuiScreens(unittest.IsolatedAsyncioTestCase):
    async def test_resources_screen_lists_via_client(self):
        fake = FakeRuntimeClient(
            agents=("default", "reviewer"), skills=("code-review",), mcp_servers=()
        )
        called: "list[str]" = []
        for m in ("list_agents", "list_skills", "list_mcp_servers"):
            _spy(fake, m, called)
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+r")
            await _wait_until(pilot, lambda: isinstance(app.screen, ResourcesScreen))
            await app.workers.wait_for_complete()
            self.assertIn("list_agents", called)
            self.assertIn("list_skills", called)
            self.assertIn("list_mcp_servers", called)
            await pilot.press("escape")  # back to chat
            await pilot.pause()
            self.assertIsInstance(app.screen, ChatScreen)

    async def test_runs_screen_lists_via_client(self):
        fake = FakeRuntimeClient()
        called: "list[str]" = []
        for m in ("list_sessions", "list_runs", "list_approvals"):
            _spy(fake, m, called)
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+o")
            await _wait_until(pilot, lambda: isinstance(app.screen, RunsScreen))
            await app.workers.wait_for_complete()
            self.assertIn("list_sessions", called)
            self.assertIn("list_runs", called)
            self.assertIn("list_approvals", called)

    async def test_doctor_screen_runs_doctor(self):
        fake = FakeRuntimeClient(doctor_report=DoctorReport())
        called: "list[str]" = []
        _spy(fake, "doctor", called)
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_doctor()
            await _wait_until(pilot, lambda: isinstance(app.screen, DoctorScreen))
            await _wait_until(pilot, lambda: "doctor" in called)
            self.assertIn("doctor", called)

    async def test_doctor_keybinding_fires_from_chat(self):
        # Regression: the composer Input is focused on the chat screen and
        # Textual's Input binds ctrl+d as delete-right; the App binding must be
        # high-priority so Ctrl+D still opens Doctor from chat.
        from textual.widgets import Input

        fake = FakeRuntimeClient()
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            self.assertTrue(pilot.app.screen.query_one(Input).has_focus)
            await pilot.press("ctrl+d")
            await _wait_until(pilot, lambda: isinstance(app.screen, DoctorScreen))
            self.assertIsInstance(app.screen, DoctorScreen)


if __name__ == "__main__":
    unittest.main()
