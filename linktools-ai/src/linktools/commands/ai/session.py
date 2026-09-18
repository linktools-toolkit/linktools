#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai session`: inspect local Runtime sessions."""

from argparse import Namespace
from typing import TYPE_CHECKING

from rich import get_console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from linktools.ai.core import Principal, PrincipalKind
from linktools.ai.runtime import (
    ListExecutionRequest,
    RuntimeHistory,
    SessionView,
)
from linktools.cli import BaseCommand

from ._common import _load_workspace, _local_runtime_state, _run_async

if TYPE_CHECKING:
    from linktools.cli import CommandParser

_DEFAULT_LIMIT = 20


class Command(BaseCommand):
    """Inspect recent sessions or one session's executions."""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("session_id", nargs="?", help="session id")

    def run(self, args: Namespace) -> int:
        workspace = _load_workspace()

        async def execute() -> int:
            async with RuntimeHistory.open(
                workspace.workspace_id,
                state=_local_runtime_state(workspace),
            ) as history:
                principal = Principal(
                    "runtime",
                    history.tenant_id,
                    PrincipalKind.LOCAL_TRUSTED.value,
                )
                if args.session_id is None:
                    _emit_sessions(
                        await history.recent_sessions(
                            principal=principal,
                            limit=_DEFAULT_LIMIT,
                        )
                    )
                else:
                    session = await history.inspect_session(
                        args.session_id,
                        principal=principal,
                    )
                    _emit_session(session)
                    await _emit_session_executions(
                        history,
                        principal,
                        session.session_id,
                    )
            return 0

        return _run_async(execute())


def _emit_sessions(sessions: tuple[SessionView, ...]) -> None:
    console = get_console()
    if not sessions:
        console.print("[dim]No sessions.[/dim]")
        return
    table = Table(title="Recent AI Sessions", box=None)
    table.add_column("Session", no_wrap=True)
    table.add_column("Status")
    table.add_column("Agent")
    table.add_column("Revision", justify="right")
    table.add_column("CWD")
    table.add_column("Active")
    table.add_column("History")
    for session in sessions:
        table.add_row(
            session.session_id,
            _status(session.status.value),
            session.agent_id,
            str(session.revision),
            session.cwd or "-",
            ", ".join(session.active_execution_ids) or "-",
            session.history_quality,
        )
    console.print(table)


def _emit_session(session: SessionView) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Session", session.session_id)
    table.add_row("Status", _status(session.status.value))
    table.add_row("Agent", session.agent_id)
    table.add_row("Revision", str(session.revision))
    table.add_row("CWD", session.cwd or "-")
    table.add_row("Active", ", ".join(session.active_execution_ids) or "-")
    table.add_row("History", session.history_quality)
    get_console().print(Panel(table, title="Session", expand=False))


async def _emit_session_executions(
    history: RuntimeHistory,
    principal: Principal,
    session_id: str,
) -> None:
    page = await history.list_executions(
        ListExecutionRequest(
            principal,
            session_id=session_id,
            limit=_DEFAULT_LIMIT,
        )
    )
    table = Table(title="Session Executions", box=None)
    table.add_column("Execution", no_wrap=True)
    table.add_column("Status")
    table.add_column("Lineage")
    table.add_column("Agent")
    table.add_column("Parent")
    for execution in page.items:
        table.add_row(
            execution.execution_id,
            _status(execution.status.value),
            execution.lineage_kind.value,
            execution.agent_id,
            execution.parent_execution_id or "-",
        )
    console = get_console()
    if page.items:
        console.print(table)
        if page.next_cursor is not None:
            console.print(f"[dim]Showing first {_DEFAULT_LIMIT} executions.[/dim]")
    else:
        console.print("[dim]No executions for this session.[/dim]")


def _status(value: str) -> Text:
    style = {
        "OPEN": "green",
        "CLOSED": "dim",
        "CLOSING": "yellow",
        "CLEANUP_REQUIRED": "red",
        "SUCCEEDED": "green",
        "FAILED": "red",
        "CANCELLED": "yellow",
    }.get(value, "")
    return Text(value, style=style)


command = Command()
