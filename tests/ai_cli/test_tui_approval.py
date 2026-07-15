#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Approval modal tests (spec §23/§32).

A run that pauses for approval pushes the approval modal; Approve resumes the
run through the runtime, Reject cancels it, Later leaves it waiting. Drives the
app with ``run_test()`` and a ``FakeRuntimeClient``."""

import types
import unittest

from linktools.ai_cli.tui.screens.chat import Composer

from linktools.ai_cli.client import FakeRuntimeClient
from linktools.ai_cli.tui.app import LinktoolsAIApp
from linktools.ai_cli.tui.modals.approval import (
    ApprovalModal,
    _mask_value,
    _render_arguments,
)
from linktools.ai_cli.tui.screens.chat import ChatScreen


async def _wait_until(pilot, cond, *, tries: int = 50) -> bool:
    for _ in range(tries):
        if cond():
            return True
        await pilot.pause()
    return cond()


def _approval() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        id="a1",
        tool_name="terminal",
        arguments={"command": ["echo", "hi"]},
        reason="high-risk shell",
    )


class TestTuiApproval(unittest.IsolatedAsyncioTestCase):
    def _fake(self) -> FakeRuntimeClient:
        return FakeRuntimeClient(
            stream_events=[{"type": "paused", "run_id": "r1", "approval_id": "a1"}],
            resume_events=[{"type": "resumed", "run_id": "r1"}],
            approval=_approval(),
        )

    async def _drive_to_modal(self, fake: FakeRuntimeClient, app, pilot) -> None:
        await pilot.pause()
        pilot.app.screen.query_one(Composer).text = "do something risky"
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, ApprovalModal))

    async def test_approve_resumes_the_run(self):
        fake = self._fake()
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await self._drive_to_modal(fake, app, pilot)
            await pilot.press("a")  # Approve
            await app.workers.wait_for_complete()
            await pilot.pause()
        self.assertEqual(fake.approve_calls, ["a1"])
        self.assertEqual(fake.resume_calls, ["r1"])

    async def test_reject_cancels_the_run(self):
        fake = self._fake()
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await self._drive_to_modal(fake, app, pilot)
            await pilot.press("r")  # Reject
            await app.workers.wait_for_complete()
            await pilot.pause()
        self.assertEqual(fake.reject_calls, [("a1", None)])
        self.assertEqual(fake.cancel_calls, ["r1"])
        self.assertEqual(fake.resume_calls, [])

    async def test_later_leaves_run_waiting(self):
        fake = self._fake()
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await self._drive_to_modal(fake, app, pilot)
            await pilot.press("l")  # Later
            await pilot.pause()
            self.assertIsInstance(app.screen, ChatScreen)
        self.assertEqual(fake.approve_calls, [])
        self.assertEqual(fake.reject_calls, [])
        self.assertEqual(fake.resume_calls, [])


class TestArgumentMasking(unittest.TestCase):
    def test_sensitive_keys_are_redacted(self):
        self.assertEqual(_mask_value("api_key", "sk-123"), "***")
        self.assertEqual(_mask_value("AUTHORIZATION", "bearer xyz"), "***")
        self.assertEqual(_mask_value("password", "hunter2"), "***")

    def test_long_values_are_truncated(self):
        value = "x" * 200
        masked = _mask_value("command", value)
        self.assertTrue(masked.endswith("…"))
        self.assertLess(len(masked), 200)

    def test_render_arguments_marks_sensitive_inline(self):
        rendered = _render_arguments({"command": ["echo"], "api_key": "sk-x"})
        self.assertIn("api_key: ***", rendered)
        self.assertIn("echo", rendered)

    def test_apikey_without_underscore_is_redacted(self):
        self.assertEqual(_mask_value("apikey", "sk-x"), "***")

    def test_nested_container_secrets_are_redacted(self):
        rendered = _render_arguments({"config": {"api_key": "sk-SECRET"}})
        self.assertNotIn("sk-SECRET", rendered)
        self.assertIn("***", rendered)


if __name__ == "__main__":
    unittest.main()
