#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Textual ChatScreen behavior tests.

Drives the app with ``run_test()`` and a ``FakeRuntimeClient`` -- no real
Runtime/model. Covers the headline flows end to end: submitting starts and
renders a streamed run, bracket-bearing model text is rendered literally (not
swallowed by Rich markup), a second submit while running is ignored, and ``Esc``
cancels the run *through the runtime* without reporting it as a failure (spec
). Textual is required, so this module imports it at top level."""

import asyncio
import unittest
from unittest import mock

from rich.markup import escape
from linktools.ai_cli.tui.screens.chat import Composer
from textual.widgets import RichLog

from linktools.ai_cli.client import FakeRuntimeClient
from linktools.ai_cli.tui.app import LinktoolsAIApp
from linktools.ai_cli.tui.messages import RunEventMessage
from linktools.ai_cli.tui.screens.chat import ChatScreen


class _BlockingClient(FakeRuntimeClient):
    """A client whose ``run_stream`` never returns, so a cancel lands mid-run."""

    async def run_stream(self, request):
        self.run_requests.append(request)
        self.last_run_id = request.run_id
        await asyncio.Event().wait()  # blocks until the worker is cancelled
        yield  # unreachable; makes this an async generator like the real client


async def _wait_until(pilot, cond, *, tries: int = 50) -> bool:
    """Let the app pump messages until ``cond()`` is true (or tries run out)."""
    for _ in range(tries):
        if cond():
            return True
        await pilot.pause()
    return cond()


def _conversation_recorder(pilot) -> "list[str]":
    """Spy on the conversation RichLog's ``write`` and return the recorder."""
    conv = pilot.app.screen.query_one("#conversation", RichLog)
    recorded: "list[str]" = []
    conv.write = recorded.append  # type: ignore[method-assign]
    return recorded


class TestTuiChatStreaming(unittest.IsolatedAsyncioTestCase):
    async def test_submit_starts_run(self):
        fake = FakeRuntimeClient(stream_events=[{"type": "text", "text": "hello"}])
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()  # ChatScreen pushed in on_mount
            pilot.app.screen.query_one(Composer).text = "hi"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
        self.assertEqual([r.prompt for r in fake.run_requests], ["hi"])
        self.assertIsNotNone(fake.last_run_id)

    async def test_submit_streams_text_to_conversation(self):
        # E2E: submit -> worker -> post_message -> dispatch -> RichLog.write.
        fake = FakeRuntimeClient(
            stream_events=[{"type": "text", "text": "hello world"}]
        )
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            recorded = _conversation_recorder(pilot)
            pilot.app.screen.query_one(Composer).text = "hi"
            await pilot.press("enter")
            await _wait_until(
                pilot, lambda: any("hello world" in str(x) for x in recorded)
            )
        self.assertTrue(
            any("hello world" in str(line) for line in recorded),
            f"streamed text not rendered: {recorded!r}",
        )

    def test_streamed_markup_chars_are_escaped(self):
        # Regression for : bracket-bearing model output must reach the
        # log escaped, not be consumed/restyled by the markup=True RichLog.
        screen = ChatScreen(client=FakeRuntimeClient())
        log = mock.MagicMock()
        with mock.patch.object(screen, "query_one", return_value=log):
            screen.on_run_event_message(
                RunEventMessage({"type": "text", "text": "[red]secret[/red]"})
            )
        log.write.assert_called_once_with(escape("[red]secret[/red]"))

    def test_render_tool_event_logs_collapsed(self):
        screen = ChatScreen(client=FakeRuntimeClient())
        log = mock.MagicMock()
        with mock.patch.object(screen, "query_one", return_value=log):
            screen.on_run_event_message(
                RunEventMessage(
                    {"type": "tool", "name": "read_file", "phase": "end", "ok": True}
                )
            )
        log.write.assert_called_once()
        self.assertIn("read_file", log.write.call_args.args[0])

    async def test_second_submit_ignored_while_running(self):
        # Spec : one active Run at a time -- a second submit is dropped.
        fake = _BlockingClient()
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            inp = pilot.app.screen.query_one(Composer)
            inp.text = "first"
            await pilot.press("enter")
            await _wait_until(pilot, lambda: fake.last_run_id is not None)
            inp.text = "second"
            await pilot.press("enter")  # run still active -> ignored
            await pilot.pause()
        self.assertEqual([r.prompt for r in fake.run_requests], ["first"])


class TestTuiChatCancel(unittest.IsolatedAsyncioTestCase):
    async def test_escape_cancels_run_through_runtime(self):
        # Spec : Esc must cancel via RuntimeClient.cancel(run_id), not only
        # stop the Worker.
        fake = _BlockingClient()
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            pilot.app.screen.query_one(Composer).text = "hi"
            await pilot.press("enter")
            # Wait until the blocking run has actually started, so the cancel
            # targets a real in-flight run_id (no start/escape race).
            await _wait_until(pilot, lambda: fake.last_run_id is not None)
            await pilot.press("escape")
            await app.workers.wait_for_complete()
        self.assertEqual(fake.cancel_calls, [fake.last_run_id])

    async def test_cancel_is_not_reported_as_failure(self):
        # A cancel raises CancelledError (BaseException), which the run worker
        # must NOT catch as a failure -> no "error:" line in the conversation.
        fake = _BlockingClient()
        app = LinktoolsAIApp(client=fake)
        async with app.run_test() as pilot:
            await pilot.pause()
            recorded = _conversation_recorder(pilot)
            pilot.app.screen.query_one(Composer).text = "hi"
            await pilot.press("enter")
            await _wait_until(pilot, lambda: fake.last_run_id is not None)
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()  # let any pending messages dispatch
        rendered = "\n".join(str(x) for x in recorded)
        self.assertNotIn("error:", rendered)


if __name__ == "__main__":
    unittest.main()
