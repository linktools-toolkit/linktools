#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai trace`: inspect the durable execution trace."""

from argparse import Namespace
from collections.abc import Mapping
from typing import TYPE_CHECKING

from rich import get_console
from rich.table import Table
from rich.text import Text

from linktools.ai.core import Principal, service_principal
from linktools.ai.runtime import RuntimeHistory
from linktools.cli import BaseCommand

from ._common import _load_workspace, _local_runtime_state, _run_async

if TYPE_CHECKING:
    from linktools.cli import CommandParser

_PAGE_LIMIT = 200


class Command(BaseCommand):
    """Inspect one execution's model/tool trace."""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("execution_id", help="execution id")

    def run(self, args: Namespace) -> int:
        workspace = _load_workspace()

        async def execute() -> int:
            async with RuntimeHistory.open(
                workspace.workspace_id,
                state=_local_runtime_state(workspace),
            ) as history:
                principal = service_principal(history.tenant_id, "ai-trace")
                await _emit_trace(history, principal, args.execution_id)
            return 0

        return _run_async(execute())


async def _emit_trace(
    history: RuntimeHistory,
    principal: Principal,
    execution_id: str,
) -> None:
    execution = await history.inspect_execution(execution_id, principal=principal)
    table = Table(
        title=(
            f"Execution Trace · {execution.execution_id} · "
            f"{execution.agent_id} · {execution.status.value}"
        ),
        box=None,
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Scope")
    table.add_column("Step", justify="right")
    table.add_column("Kind")
    table.add_column("Status")
    table.add_column("Detail", overflow="fold")
    table.add_column("Duration", justify="right")

    cursor: str | None = None
    while True:
        page = await history.trace(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        for item in page.items:
            payload = item.payload if isinstance(item.payload, Mapping) else {}
            table.add_row(
                str(item.sequence),
                _scope(payload),
                _value(payload.get("step_index")),
                _value(payload.get("kind")),
                _status(payload.get("status")),
                _detail(payload),
                _duration(payload.get("duration_ns")),
            )
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    get_console().print(table)


def _scope(payload: Mapping[object, object]) -> str:
    value = payload.get("scope")
    segment = payload.get("segment_sequence")
    if value is None:
        return "-"
    return f"{value}/{segment}" if segment is not None else str(value)


def _detail(payload: Mapping[object, object]) -> str:
    tool = payload.get("tool_name")
    if isinstance(tool, str) and tool:
        return tool
    request = payload.get("request_sequence")
    purpose = payload.get("purpose")
    usage = payload.get("token_usage")
    parts: list[str] = []
    if request is not None:
        parts.append(f"request #{request}")
    if isinstance(purpose, str) and purpose:
        parts.append(purpose)
    if isinstance(usage, Mapping):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            parts.append(f"{input_tokens:,} in / {output_tokens:,} out")
    child = payload.get("child_execution_id")
    if isinstance(child, str) and child:
        parts.append(f"child={child}")
    return " · ".join(parts) or "-"


def _status(value: object) -> Text:
    text = _value(value)
    style = {
        "SUCCEEDED": "green",
        "FAILED": "red",
        "STARTED": "cyan",
    }.get(text, "")
    return Text(text, style=style)


def _duration(value: object) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return "-"
    seconds = value / 1_000_000_000
    if seconds >= 1:
        return f"{seconds:.3f}s"
    milliseconds = value / 1_000_000
    if milliseconds >= 1:
        return f"{milliseconds:.3f}ms"
    microseconds = value / 1_000
    if microseconds >= 1:
        return f"{microseconds:.3f}us"
    return f"{value}ns"


def _value(value: object) -> str:
    return "-" if value is None else str(value)


command = Command()
