#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI contract tests for `lt ai run` / `chat` / `resume` (spec §22/§23/§24).

Covers the behaviours the spec names explicitly:

* ``run`` loads the PROJECT agent (build_project_bundle) and exits 4 on pause,
  130 + cancel on Ctrl+C (§22);
* ``chat`` cancels the in-flight Run through the runtime on Ctrl+C (§23) and
  wires the interactive Approve/Reject/Later flow (§24);
* ``resume`` restores the run via the project bundle, passing ONLY the run id
  (§24), and re-exits 4 if it pauses again;
* ``announce_paused`` surfaces tool/arguments/reason/run_id/approval_id (§24).
"""

import argparse
import asyncio
import types
import unittest
from pathlib import Path
from unittest import mock

from linktools.commands.ai import chat, resume, run
from linktools.commands.ai.support import announce_paused


class _FakeApprovalRequest:
    def __init__(self, tool_name, arguments, reason):
        self.tool_name = tool_name
        self.arguments = arguments
        self.reason = reason


class _FakeApprovals:
    def __init__(self, request=None):
        self._request = request

    async def get(self, approval_id):
        return self._request


class _FakeStorage:
    def __init__(self, request=None):
        self.approvals = _FakeApprovals(request)


class _FakeRuntime:
    """Minimal runtime double: streaming + cancel + resume, all recorded."""

    def __init__(self, stream_events=None, resume_events=None, stream_error=None):
        self._stream_events = stream_events or []
        self._resume_events = resume_events or []
        self._stream_error = stream_error
        self.cancel = mock.AsyncMock()
        self.last_stream_run_id = None
        self.resume_calls: "list[str]" = []

    async def run_stream(self, spec, prompt, *, session_id=None, run_id=None):
        self.last_stream_run_id = run_id
        for event in self._stream_events:
            yield event
        if self._stream_error is not None:
            raise self._stream_error

    async def resume(self, run_id):
        self.resume_calls.append(run_id)
        for event in self._resume_events:
            yield event


def _bundle(runtime, storage=None):
    return types.SimpleNamespace(runtime=runtime, storage=storage or _FakeStorage())


class _RecordingLogger:
    def __init__(self):
        self.lines: "list[str]" = []

    def warning(self, message, *args):
        self.lines.append(message % args if args else message)

    def info(self, message, *args):
        self.lines.append(message % args if args else message)


async def _noop(*args, **kwargs):
    return None


def _args(**kwargs):
    return argparse.Namespace(
        model="m",
        base_url="https://x",
        api_key="k",
        workdir=None,
        session="main",
        agent=None,
        **kwargs,
    )


class TestAiRunExitCodes(unittest.TestCase):
    """`lt ai run` loads the project agent and emits the §22 exit codes."""

    def _patches(self, bundle_obj):
        return [
            mock.patch(
                "linktools.commands.ai.run.build_project_bundle",
                lambda args: bundle_obj,
            ),
            mock.patch("linktools.commands.ai.run.load_agent_spec", _noop),
            mock.patch("linktools.commands.ai.run.ensure_session", _noop),
            mock.patch("linktools.commands.ai.run.announce_paused", _noop),
        ]

    def test_success_returns_zero(self):
        fake = _FakeRuntime(stream_events=[{"type": "text", "text": "hello"}])
        patches = self._patches(_bundle(fake))
        for patch in patches:
            patch.start()
        try:
            code = run.Command().run(_args(prompt="hi"))
        finally:
            for patch in patches:
                patch.stop()
        self.assertEqual(code, 0)

    def test_paused_returns_exit_code_4(self):
        fake = _FakeRuntime(
            stream_events=[{"type": "paused", "run_id": "r1", "approval_id": "a1"}]
        )
        patches = self._patches(_bundle(fake))
        for patch in patches:
            patch.start()
        try:
            code = run.Command().run(_args(prompt="hi"))
        finally:
            for patch in patches:
                patch.stop()
        self.assertEqual(code, 4)

    def test_interrupt_returns_130_and_cancels_run(self):
        fake = _FakeRuntime(
            stream_events=[{"type": "text", "text": "partial"}],
            stream_error=asyncio.CancelledError(),
        )
        patches = self._patches(_bundle(fake))
        for patch in patches:
            patch.start()
        try:
            code = run.Command().run(_args(prompt="hi"))
        finally:
            for patch in patches:
                patch.stop()
        self.assertEqual(code, 130)
        fake.cancel.assert_awaited_once_with(fake.last_stream_run_id)


class TestAiChatCancel(unittest.TestCase):
    """`lt ai chat` must cancel the in-flight Run on Ctrl+C (§23)."""

    def test_interrupt_cancels_in_flight_run(self):
        fake = _FakeRuntime(
            stream_events=[{"type": "text", "text": "x"}],
            stream_error=asyncio.CancelledError(),
        )
        cmd = chat.Command()
        with mock.patch("linktools.commands.ai.chat.new_run_id", lambda: "fixed-id"):
            asyncio.run(cmd._run_turn(fake, object(), _FakeStorage(), "s", "line"))
        fake.cancel.assert_awaited_once_with("fixed-id")


class TestAiChatInteractiveApproval(unittest.TestCase):
    """`lt ai chat` interactive Approve/Reject/Later wiring (§24)."""

    def _event(self):
        return {"type": "paused", "run_id": "r1", "approval_id": "a1"}

    def _drive(self, input_value, fake):
        resolve = mock.AsyncMock(return_value=0)
        with (
            mock.patch("builtins.input", return_value=input_value),
            mock.patch("linktools.commands.ai.chat.announce_paused", _noop),
            mock.patch("linktools.commands.ai.chat.resolve_approval", resolve),
        ):
            asyncio.run(
                chat.Command()._handle_paused(fake, _FakeStorage(), self._event())
            )
        return resolve

    def test_approve_resolves_then_resumes(self):
        fake = _FakeRuntime(
            resume_events=[
                {"type": "resumed", "run_id": "r1"},
                {"type": "text", "text": "ok"},
            ]
        )
        resolve = self._drive("approve", fake)
        resolve.assert_awaited_once()
        _, kwargs = resolve.call_args
        self.assertTrue(kwargs["approved"])
        self.assertEqual(fake.resume_calls, ["r1"])
        fake.cancel.assert_not_awaited()

    def test_reject_resolves_then_cancels(self):
        fake = _FakeRuntime()
        resolve = self._drive("reject", fake)
        _, kwargs = resolve.call_args
        self.assertFalse(kwargs["approved"])
        fake.cancel.assert_awaited_once_with("r1")
        self.assertEqual(fake.resume_calls, [])

    def test_later_keeps_pending_without_resolve_or_resume(self):
        fake = _FakeRuntime()
        resolve = self._drive("later", fake)
        resolve.assert_not_awaited()
        fake.cancel.assert_not_awaited()
        self.assertEqual(fake.resume_calls, [])

    def test_blank_input_defaults_to_later(self):
        fake = _FakeRuntime()
        resolve = self._drive("", fake)
        resolve.assert_not_awaited()
        self.assertEqual(fake.resume_calls, [])
        fake.cancel.assert_not_awaited()


class TestAiResume(unittest.TestCase):
    """`lt ai resume` restores by run id only (§24), via the project bundle."""

    def _patches(self, bundle_obj):
        return [
            mock.patch(
                "linktools.commands.ai.resume.build_project_bundle",
                lambda args: bundle_obj,
            ),
            mock.patch("linktools.commands.ai.resume.announce_paused", _noop),
        ]

    def test_resume_passes_only_run_id_no_spec(self):
        fake = _FakeRuntime(
            resume_events=[
                {"type": "resumed", "run_id": "r1"},
                {"type": "text", "text": "done"},
            ]
        )
        patches = self._patches(_bundle(fake))
        for patch in patches:
            patch.start()
        try:
            code = resume.Command().run(_args(run_id="r1"))
        finally:
            for patch in patches:
                patch.stop()
        self.assertEqual(code, 0)
        self.assertEqual(fake.resume_calls, ["r1"])

    def test_resume_paused_again_returns_exit_code_4(self):
        fake = _FakeRuntime(
            resume_events=[
                {"type": "resumed", "run_id": "r1"},
                {"type": "paused", "run_id": "r1", "approval_id": "a2"},
            ]
        )
        patches = self._patches(_bundle(fake))
        for patch in patches:
            patch.start()
        try:
            code = resume.Command().run(_args(run_id="r1"))
        finally:
            for patch in patches:
                patch.stop()
        self.assertEqual(code, 4)


class TestAnnouncePaused(unittest.TestCase):
    """`announce_paused` renders the §24 fields from the stored request."""

    def test_renders_tool_arguments_reason_and_ids(self):
        request = _FakeApprovalRequest(
            tool_name="terminal",
            arguments={"command": ["echo", "hi"]},
            reason="high-risk shell",
        )
        storage = _FakeStorage(request=request)
        logger = _RecordingLogger()
        event = {"type": "paused", "run_id": "r1", "approval_id": "a1"}

        asyncio.run(announce_paused(storage, event, logger))

        rendered = "\n".join(logger.lines)
        self.assertIn("tool: terminal", rendered)
        self.assertIn("'command': ['echo', 'hi']", rendered)
        self.assertIn("reason: high-risk shell", rendered)
        self.assertIn("run_id: r1", rendered)
        self.assertIn("approval_id: a1", rendered)
        self.assertIn("lt ai approve a1", rendered)
        self.assertIn("lt ai resume r1", rendered)

    def test_missing_request_degrades_to_ids_only(self):
        storage = _FakeStorage(request=None)
        logger = _RecordingLogger()
        event = {"type": "paused", "run_id": "r9", "approval_id": "a9"}

        asyncio.run(announce_paused(storage, event, logger))

        rendered = "\n".join(logger.lines)
        self.assertIn("tool: ?", rendered)
        self.assertIn("run_id: r9", rendered)
        self.assertIn("approval_id: a9", rendered)


class TestApproveUsesProjectStorage(unittest.TestCase):
    """`lt ai approve` resolves against the PROJECT's storage (where a project
    run's paused approval lives), not the global ai data dir (§24 cross-process)."""

    def test_approve_uses_project_storage(self):
        from linktools.commands.ai import approve

        fake_storage = _FakeStorage()
        resolve = mock.AsyncMock(return_value=0)
        with (
            mock.patch(
                "linktools.commands.ai.approve.project_storage", lambda: fake_storage
            ),
            mock.patch("linktools.commands.ai.approve.resolve_approval", resolve),
        ):
            code = approve.Command().run(_args(approval_id="a1"))
        self.assertEqual(code, 0)
        resolve.assert_awaited_once()
        self.assertIs(resolve.call_args.args[0], fake_storage)


class TestChatSlashCommands(unittest.IsolatedAsyncioTestCase):
    """`lt ai chat` slash-command dispatch (§23)."""

    def _bundle(self):
        return types.SimpleNamespace(
            project=types.SimpleNamespace(default_agent="default"),
            storage=_FakeStorage(),
            runtime=object(),
        )

    async def test_agent_command_reloads_spec(self):
        cmd = chat.Command()
        with (
            mock.patch(
                "linktools.commands.ai.chat.load_agent_spec",
                new_callable=mock.AsyncMock,
            ) as load,
            mock.patch("linktools.commands.ai.chat.ensure_session", _noop),
        ):
            _, _, stop = await cmd._handle_command(
                self._bundle(), "OLD_SPEC", "main", "/agent reviewer"
            )
        self.assertFalse(stop)
        load.assert_awaited_once()
        self.assertEqual(load.call_args.args[1], "reviewer")

    async def test_new_command_switches_session(self):
        cmd = chat.Command()
        with mock.patch("linktools.commands.ai.chat.ensure_session", _noop):
            _, session_id, stop = await cmd._handle_command(
                self._bundle(), "spec", "old", "/new demo"
            )
        self.assertEqual(session_id, "demo")
        self.assertFalse(stop)

    async def test_exit_command_stops_repl(self):
        cmd = chat.Command()
        _, _, stop = await cmd._handle_command(object(), "spec", "main", "/exit")
        self.assertTrue(stop)

    async def test_unknown_command_does_not_stop(self):
        cmd = chat.Command()
        _, _, stop = await cmd._handle_command(self._bundle(), "spec", "main", "/nope")
        self.assertFalse(stop)

    async def test_inspect_command_calls_runtime_inspect(self):
        cmd = chat.Command()
        bundle = types.SimpleNamespace(
            project=types.SimpleNamespace(default_agent="default"),
            storage=_FakeStorage(),
            runtime=types.SimpleNamespace(
                inspect=mock.AsyncMock(
                    return_value=types.SimpleNamespace(tool_descriptors=(), warnings=())
                )
            ),
        )
        await cmd._handle_command(bundle, "SPEC", "main", "/inspect")
        bundle.runtime.inspect.assert_awaited_once_with("SPEC")


class TestChatCommandErrorHandling(unittest.IsolatedAsyncioTestCase):
    """A failing slash command (e.g. /resume <stale_id>) must not kill the REPL."""

    async def test_failing_resume_returns_to_prompt(self):
        from linktools.ai.errors import RunNotFoundError

        class _RT:
            async def resume(self, run_id):
                raise RunNotFoundError(f"no run: {run_id}")

            async def cancel(self, run_id):
                return None

        bundle = types.SimpleNamespace(
            project=types.SimpleNamespace(default_agent="default", root=Path(".")),
            storage=_FakeStorage(),
            runtime=_RT(),
        )
        cmd = chat.Command()
        inputs = iter(["/resume stale-id", "/exit"])
        with (
            mock.patch(
                "linktools.commands.ai.chat.build_project_bundle", lambda args: bundle
            ),
            mock.patch("linktools.commands.ai.chat.load_agent_spec", mock.AsyncMock()),
            mock.patch("linktools.commands.ai.chat.ensure_session", _noop),
            mock.patch("builtins.input", lambda *a, **k: next(inputs)),
        ):
            code = await cmd._chat_async(
                types.SimpleNamespace(
                    session="main", model=None, base_url=None, api_key=None
                )
            )
        # Reached /exit cleanly (0); the failing /resume did NOT crash the loop.
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
