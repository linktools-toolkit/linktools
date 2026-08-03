#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The workspace screen: the persistent Claude-Code-style layout.

A single screen that composes three panes -- :class:`Sidebar` (left),
conversation area (center), :class:`ContextPanel` (right) -- plus a
:class:`StatusBar` and :class:`Composer`. This replaces the previous
push_screen overlay design where Catalog/Runs/Doctor each took over the whole
screen; all of that state now lives in the persistent side panels.

One active run at a time. Submitting starts a ``@work`` worker that streams
``RuntimeClient.run_stream`` events into the :class:`Conversation`; the
conversation view re-renders on each event. ``Esc``/``Ctrl+C`` cancel the run
through the worker AND through ``RuntimeClient.cancel`` (cancelling the worker
alone doesn't stop a suspended model call). A pause event opens the approval
modal."""

from typing import TYPE_CHECKING

import asyncio

from textual import work
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import Screen
from textual.widgets import Footer, Header

from ..conversation import Conversation
from ..messages import RunEventMessage, RunFailedMessage, RunFinishedMessage
from ..modals.approval import ApprovalModal
from ..widgets import (
    Composer,
    ContextPanel,
    ConversationView,
    SessionSelected,
    Sidebar,
    StatusBar,
)
from ...client import RunRequest, RuntimeClient, new_run_id

if TYPE_CHECKING:
    from textual.app import ComposeResult


async def _client_model_id(client: "RuntimeClient") -> "str | None":
    """Best-effort primary-model id from a local client's default agent.

    Remote/fake clients have no bundle; the panel keeps "(unset)" for them.
    A failure to resolve degrades to None rather than breaking the refresh."""
    bundle = getattr(client, "bundle", None)
    if bundle is None:
        return None
    try:
        spec = await bundle.agents.get(bundle.project.default_agent)
        return getattr(getattr(spec, "model", None), "primary", None)
    except Exception:
        return None


class WorkspaceScreen(Screen):
    """The persistent three-pane workspace."""

    BINDINGS = [
        Binding("escape", "cancel_run", "Cancel", show=False),
        Binding("ctrl+c", "cancel_run", "Cancel", show=False),
        Binding("ctrl+l", "clear_conversation", "Clear"),
        Binding("ctrl+n", "new_session", "New session"),
        Binding("ctrl+q", "app.quit", "Quit"),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self, client: "RuntimeClient") -> None:
        super().__init__()
        self.client = client
        self.conversation = Conversation()
        self.session_id = "main"
        self._active_run_id: "str | None" = None
        self._active_worker = None
        # The run id of a pause whose approval modal is open (or was dismissed
        # with "Later"). Stays set so a second prompt in the same session
        # warns rather than silently overlapping a paused run.
        self._pending_approval_run_id: "str | None" = None
        self._refresh_sidebar_lock = asyncio.Lock()

    def compose(self) -> "ComposeResult":
        yield Header(show_clock=False)
        yield Horizontal(
            Sidebar(id="sidebar"),
            ConversationView(id="conversation"),
            ContextPanel(id="context"),
            id="workspace-body",
        )
        yield StatusBar(id="status")
        yield Composer(id="composer")
        yield Footer()

    def action_help(self) -> None:
        from ..modals.help import HelpModal

        self.app.push_screen(HelpModal())

    def on_mount(self) -> None:
        self.query_one(ConversationView).bind(self.conversation)
        self.query_one(Sidebar).set_active_session(self.session_id)
        self._refresh_status()
        self.query_one(Composer).focus_input()
        self.run_worker(self.refresh_sidebar(), group="refresh-sidebar")
        self.run_worker(self.refresh_context(), group="refresh-context")

    # -- public refresh hooks (app + tests drive these) ------------------ #

    async def refresh_sidebar(self) -> None:
        # Serialize against concurrent on_mount / on_run_finished refreshes so
        # two refreshes never interleave their clear/append on the ListView
        # (which would stack duplicate session-row ids).
        async with self._refresh_sidebar_lock:
            sidebar = self.query_one(Sidebar)
            client = self.client
            sessions = await self._safe(client.list_sessions, default=[])
            runs = await self._safe(client.list_runs, default=[])
            approvals = await self._safe(client.list_approvals, default=[])
            agents = await self._safe(client.list_agents, default=())
            skills = await self._safe(client.list_skills, default=())
            mcp = await self._safe(client.list_mcp_servers, default=())
            sidebar.render_sessions(sessions, active=self.session_id)
            sidebar.render_runs(runs)
            sidebar.render_approvals(approvals)
            sidebar.render_resources(agents=agents, skills=skills, mcp=mcp)

    async def _safe(self, coro_fn, *, default):
        """Run a sidebar/context listing call; a transient backend failure
        degrades to the empty default (with a toast) rather than aborting the
        whole refresh."""
        try:
            return await coro_fn()
        except Exception as exc:
            self.app.notify(
                f"{getattr(coro_fn, '__name__', 'listing')} failed: {exc}",
                title="Refresh",
                severity="warning",
            )
            return default

    async def refresh_context(self) -> None:
        panel = self.query_one(ContextPanel)
        client = self.client
        inspection = await self._safe(lambda: client.inspect(None), default=None)
        panel.render_inspection(inspection, agent_id=None)
        # Doctor summary: cheap local checks, run once per refresh so the panel
        # reflects the current state (a failed check is visible immediately,
        # not only when the user opens the modal).
        doctor = await self._safe(client.doctor, default=None)
        panel.render_doctor(doctor)
        # Model: pull the agent's primary model id from the bundle when the
        # client exposes it (LocalRuntimeClient does; FakeRuntimeClient does
        # not, in which case the panel keeps "(unset)").
        model_id = await _client_model_id(client)
        panel.render_model(model_id)

    # -- submit ----------------------------------------------------------- #

    def on_composer_submitted(self, event: Composer.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        if text.startswith("/"):
            from ..commands import handle_slash_command

            handle_slash_command(self, text)
            return
        if self._active_worker is not None:
            self.conversation.add_status(
                "(a run is already in progress)", tone="yellow"
            )
            self._refresh_view()
            self._refocus_composer()
            return
        if self._pending_approval_run_id is not None:
            self.conversation.add_status(
                f"(run {self._pending_approval_run_id[:8]}… is awaiting approval; "
                "resolve it before starting a new run)",
                tone="yellow",
            )
            self._refresh_view()
            self._refocus_composer()
            return
        self.conversation.add_user(text)
        run_id = new_run_id()
        self._active_run_id = run_id
        self._refresh_view()
        self._refresh_status()
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
        self.conversation.apply(event)
        self._refresh_view()

    def on_run_finished_message(self, message: RunFinishedMessage) -> None:
        self._refresh_status()
        self.run_worker(self.refresh_sidebar(), group="refresh-sidebar")
        self._refocus_composer()

    def on_run_failed_message(self, message: RunFailedMessage) -> None:
        self.conversation.add_error(str(message.error))
        self._refresh_view()
        self._refresh_status()
        self._refocus_composer()

    # -- sidebar session selection --------------------------------------- #

    def on_session_selected(self, event: SessionSelected) -> None:
        from ...client import validate_session_id

        try:
            self.session_id = validate_session_id(event.session_id)
        except Exception as exc:
            self.conversation.add_status(f"invalid session: {exc}", tone="red")
            self._refresh_view()
            return
        # Load the selected session's prior turns into the conversation so the
        # user sees real history, not just a switched label. Runs in a worker
        # so a slow backend doesn't block the UI.
        self.run_worker(
            self._load_session_history(self.session_id), group="load-history"
        )

    async def _load_session_history(self, session_id: str) -> None:
        per_turn = await self._safe(
            lambda: self.client.get_session_messages(session_id), default=()
        )
        self.conversation.clear()
        self.conversation.load_from_turns(per_turn)
        turn_count = len(per_turn)
        if turn_count:
            self.conversation.add_status(
                f"loaded session {session_id} ({turn_count} turn(s))", tone="dim"
            )
        else:
            self.conversation.add_status(f"session {session_id} (empty)", tone="dim")
        self._refresh_view()
        self.query_one(Sidebar).set_active_session(session_id)
        self._refresh_status()
        self._refocus_composer()

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
            self.app.notify(f"cancel failed: {exc}", title="Runtime", severity="error")
            from linktools.core import environ

            environ.logger.warning("runtime cancel failed for %s: %s", run_id, exc)
        finally:
            self._refocus_composer()

    # -- approval -------------------------------------------------------- #

    def _open_approval(self, event) -> None:
        run_id = event.get("run_id")
        approval_id = event.get("approval_id")
        # The run worker that produced the pause has finished; clear the
        # active-worker slot so the state is honest, but keep the run marked
        # pending-approval so a second submit warns instead of overlapping.
        self._active_worker = None
        self._active_run_id = None
        self._pending_approval_run_id = run_id
        self._refresh_status()

        def _on_decision(decision: "str | None") -> None:
            if decision == "approve":
                self._pending_approval_run_id = None
                self._active_run_id = run_id
                self._active_worker = self._start_approval(approval_id, run_id)
            elif decision == "reject":
                self._pending_approval_run_id = None
                self._start_reject(approval_id, run_id)
            # "later" (None): leave _pending_approval_run_id set so a new
            # submit warns about the unresolved approval. The modal is gone,
            # so refocus the composer for continued typing.

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
        try:
            await self.client.reject(approval_id)
            await self.client.cancel(run_id)
            self.conversation.add_status("rejected and cancelled", tone="red")
        except Exception as exc:
            self.conversation.add_error(f"reject failed: {exc}")
        self._refresh_view()

    # -- clear / new session --------------------------------------------- #

    def action_clear_conversation(self) -> None:
        self.conversation.clear()
        self._refresh_view()

    def action_new_session(self) -> None:
        from ...client import validate_session_id

        new_id = new_run_id().replace("-", "")[:12]
        try:
            self.session_id = validate_session_id(new_id)
        except Exception:
            self.session_id = "main"
        self.conversation.clear()
        self.conversation.add_status(f"new session {self.session_id}")
        self._refresh_view()
        self.query_one(Sidebar).set_active_session(self.session_id)
        self._refresh_status()

    # -- helpers ---------------------------------------------------------- #

    def _refresh_view(self) -> None:
        self.query_one(ConversationView).refresh_from()

    def _refresh_status(self) -> None:
        bar = self.query_one(StatusBar)
        composer = self.query_one(Composer)
        if self._active_run_id is not None:
            bar.set_status(
                f"session: {self.session_id} · running {self._active_run_id[:8]}…"
            )
            composer.set_helper("running…  Esc/Ctrl+C to cancel")
        elif self._pending_approval_run_id is not None:
            bar.set_status(
                f"session: {self.session_id} · approval pending "
                f"({self._pending_approval_run_id[:8]}…)"
            )
            composer.set_helper("approval pending — resolve it before a new run")
        else:
            bar.set_session(self.session_id)
            composer.set_helper(
                "Enter to send · Shift+Enter for newline · /help for commands"
            )

    def _refocus_composer(self) -> None:
        # Re-focus the composer input after any state change that could have
        # moved focus away (modal close, submit, cancel). A missing composer
        # during teardown is ignored.
        try:
            self.query_one(Composer).focus_input()
        except Exception:
            pass
