#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The chat screen: composer + streaming conversation.

One active run at a time. Submitting starts a ``@work`` worker that streams
``RuntimeClient.run_stream`` events onto the conversation. ``Esc``/``Ctrl+C``
cancel the run -- cancelling the Worker alone is not enough, so the run is also
cancelled through ``RuntimeClient.cancel(run_id)``.

Untrusted text is Rich-markup-escaped before writing."""

from rich.markup import escape
from textual import work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.message import Message
from textual.screen import Screen
from textual.widgets import RichLog, TextArea

from linktools.core import environ

from ..messages import RunEventMessage, RunFailedMessage, RunFinishedMessage
from ..modals.approval import ApprovalModal
from ...client import RunRequest, RuntimeClient, new_run_id


class Composer(TextArea):
    """Multi-line text area that submits on Enter (Shift+Enter for newline)."""

    BINDINGS = [
        Binding("enter", "submit", "Send", show=False, priority=True),
    ]

    def action_submit(self) -> None:
        self.post_message(Composer.Submitted(self))

    class Submitted(Message):
        def __init__(self, composer: "Composer") -> None:
            super().__init__()
            self.composer = composer
            self.value = composer.text


class ChatScreen(Screen):
    """Conversation + composer. Talks to the backend only via RuntimeClient."""

    BINDINGS = [
        Binding("escape", "cancel_run", "Cancel", show=False),
        Binding("ctrl+c", "cancel_run", "Cancel", show=False),
        Binding("ctrl+q", "app.quit", "Quit"),
    ]

    def __init__(self, client: "RuntimeClient") -> None:
        super().__init__()
        self.client = client
        self.session_id = "main"
        self._active_run_id: "str | None" = None
        self._active_worker = None

    def compose(self) -> ComposeResult:
        yield RichLog(id="conversation", wrap=True, markup=True)
        yield Composer(id="composer")

    def on_mount(self) -> None:
        self.query_one("#composer", Composer).focus()

    # -- submit ----------------------------------------------------------- #

    def on_composer_submitted(self, event: Composer.Submitted) -> None:
        text = event.value.strip()
        if not text:
            event.composer.text = ""
            return
        event.composer.text = ""
        if text.startswith("/"):
            from ..commands import handle_slash_command

            handle_slash_command(self, text)
            return
        if self._active_worker is not None:
            return
        self.query_one("#conversation", RichLog).write(f"[b]you[/b]: {escape(text)}")
        run_id = new_run_id()
        self._active_run_id = run_id
        self._active_worker = self._start_run(text, run_id)

    @work(exclusive=True, group="active-run")
    async def _start_run(self, prompt: str, run_id: str) -> None:
        request = RunRequest(prompt=prompt, session_id=self.session_id, run_id=run_id)
        try:
            async for event in self.client.run_stream(request):
                self.post_message(RunEventMessage(event))
            self.post_message(RunFinishedMessage(run_id))
        except Exception as exc:  # CancelledError is BaseException: not caught
            self.post_message(RunFailedMessage(exc))
        finally:
            self._active_run_id = None
            self._active_worker = None

    # -- messages --------------------------------------------------------- #

    def on_run_event_message(self, message: RunEventMessage) -> None:
        event = message.event
        if event.get("type") == "paused":
            self._open_approval(event)
            return
        log = self.query_one("#conversation", RichLog)
        kind = event.get("type")
        if kind == "text":
            log.write(escape(event.get("text", "")))
        elif kind == "tool":
            ok = " ok" if event.get("ok") else ""
            name = escape(str(event.get("name", "")))
            phase = escape(str(event.get("phase", "")))
            log.write(f"[dim]\\[tool: {name} {phase}{ok}][/dim]")
        elif kind == "failed":
            # The Outcome model (spec 12.3) reports run failure as a stream
            # event rather than a raised exception -- render it the same way
            # ``on_run_failed_message`` renders a genuinely raised error.
            error_type = escape(str(event.get("error_type", "")))
            error_message = escape(str(event.get("message", "")))
            log.write(f"[red]error: {error_type}: {error_message}[/red]")
        elif kind == "cancelled":
            log.write("[yellow]run cancelled[/yellow]")

    def on_run_finished_message(self, message: RunFinishedMessage) -> None:
        pass

    def on_run_failed_message(self, message: RunFailedMessage) -> None:
        self.query_one("#conversation", RichLog).write(
            f"[red]error: {escape(str(message.error))}[/red]"
        )

    # -- cancel ----------------------------------------------------------- #

    def action_cancel_run(self) -> None:
        worker = self._active_worker
        run_id = self._active_run_id
        if worker is None or run_id is None:
            return
        self._active_worker = None
        self._active_run_id = None
        worker.cancel()
        self._cancel_via_runtime(run_id)

    @work(group="cancel-run")
    async def _cancel_via_runtime(self, run_id: str) -> None:
        try:
            await self.client.cancel(run_id)
        except Exception as exc:
            environ.logger.warning("runtime cancel failed for %s: %s", run_id, exc)

    # -- approval -------------------------------------------------------- #

    def _open_approval(self, event) -> None:
        def _on_decision(decision: "str | None") -> None:
            run_id = event.get("run_id")
            approval_id = event.get("approval_id")
            if decision == "approve":
                self._active_run_id = run_id
                self._active_worker = self._start_approval(approval_id, run_id)
            elif decision == "reject":
                self._start_reject(approval_id, run_id)

        self.app.push_screen(
            ApprovalModal(client=self.client, event=event), _on_decision
        )

    @work(exclusive=True, group="active-run")
    async def _start_approval(self, approval_id: "str | None", run_id: str) -> None:
        try:
            await self.client.approve(approval_id)
            async for event in self.client.resume_stream(run_id):
                self.post_message(RunEventMessage(event))
        except Exception as exc:
            self.post_message(RunFailedMessage(exc))
        finally:
            self._active_run_id = None
            self._active_worker = None

    @work(group="approval")
    async def _start_reject(self, approval_id: "str | None", run_id: str) -> None:
        log = self.query_one("#conversation", RichLog)
        try:
            await self.client.reject(approval_id)
            await self.client.cancel(run_id)
            log.write("[red]rejected and cancelled[/red]")
        except Exception as exc:
            environ.logger.warning("reject failed for %s: %s", run_id, exc)
            log.write(f"[red]reject failed: {escape(str(exc))}[/red]")
