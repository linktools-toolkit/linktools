#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai history`: inspect persisted local Runtime execution history."""

import json
from argparse import Namespace
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from rich import get_console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from linktools.ai.core import Principal, UsageMetrics, service_principal
from linktools.ai.runtime import (
    ExecutionInfo,
    ModelInteractionItem,
    Page,
    RuntimeHistory,
)
from linktools.cli import BaseCommand

from ._common import _load_workspace, _local_runtime_state, _run_async

if TYPE_CHECKING:
    from linktools.cli import CommandParser

_PAGE_LIMIT = 200
_DEFAULT_LIST_LIMIT = 20
_PREVIEW_LIMIT = 120


class Command(BaseCommand):
    """Inspect local AI execution history."""

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("execution_id", nargs="?", help="execution id")

    def run(self, args: Namespace) -> int:
        workspace = _load_workspace()

        async def execute() -> int:
            async with RuntimeHistory.open(
                workspace.workspace_id,
                state=_local_runtime_state(workspace),
            ) as history:
                principal = service_principal(history.tenant_id, "ai-history")
                if args.execution_id is None:
                    _emit_execution_list(
                        await history.recent_executions(
                            principal=principal,
                            limit=_DEFAULT_LIST_LIMIT,
                        )
                    )
                else:
                    await _emit_execution_detail(history, principal, args.execution_id)
            return 0

        return _run_async(execute())


def _emit_execution_list(executions: tuple[ExecutionInfo, ...]) -> None:
    console = get_console()
    if not executions:
        console.print("[dim]No executions.[/dim]")
        return

    table = Table(title="Recent AI Executions", box=None)
    table.add_column("Created", style="dim", no_wrap=True)
    table.add_column("Execution", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Agent")
    table.add_column("Session")
    table.add_column("Parent")
    table.add_column("Error")
    for execution in executions:
        table.add_row(
            execution.created_at.isoformat(timespec="seconds"),
            execution.execution_id,
            _status_text(execution.status.value),
            execution.agent_id,
            execution.session_id or "-",
            execution.parent_execution_id or "-",
            execution.error_code or "-",
        )
    console.print(table)


async def _emit_execution_detail(
    history: RuntimeHistory,
    principal: Principal,
    execution_id: str,
) -> None:
    console = get_console()
    execution = await history.inspect_execution(execution_id, principal=principal)
    console.print(_execution_panel(execution))

    first_model_page = await history.model_interactions(
        execution_id,
        principal=principal,
        cursor=None,
        limit=_PAGE_LIMIT,
    )
    representative = next(
        (
            item
            for item in first_model_page.items
            if item.depth == 0 and item.purpose == "agent"
        ),
        first_model_page.items[0] if first_model_page.items else None,
    )
    if representative is None:
        console.print(Panel("[dim]No model requests.[/dim]", title="Prompt Architecture"))
    else:
        console.print(_prompt_architecture_tree(representative))

    await _emit_history_rows(history, principal, execution_id)
    await _emit_transcript_rows(history, principal, execution_id)
    await _emit_model_interactions(
        history,
        principal,
        execution_id,
        first_model_page,
    )


def _execution_panel(execution: ExecutionInfo) -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("Execution", execution.execution_id)
    table.add_row("Status", _status_text(execution.status.value))
    table.add_row("Agent", execution.agent_id)
    table.add_row("Session", execution.session_id or "-")
    table.add_row("Lineage", execution.lineage_kind.value)
    table.add_row("Parent", execution.parent_execution_id or "-")
    table.add_row("Created", execution.created_at.isoformat())
    table.add_row("Updated", execution.updated_at.isoformat())
    table.add_row("Error", execution.error_code or "-")
    if execution.safe_error_details:
        table.add_row("Safe details", _preview(execution.safe_error_details))
    if execution.error_diagnostics is not None:
        table.add_row(
            "Diagnostics",
            f"{execution.error_diagnostics.exception_type}: "
            f"{_preview(execution.error_diagnostics.exception_message)}",
        )
    return Panel(table, title="Execution", expand=False)


async def _emit_history_rows(
    history: RuntimeHistory,
    principal: Principal,
    execution_id: str,
) -> None:
    console = get_console()
    cursor: str | None = None
    emitted = False
    first = True
    while True:
        page = await history.history(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        if page.items:
            table = Table(
                title="Conversation History" if first else None,
                box=None,
                show_header=first,
            )
            table.add_column("#", justify="right", style="dim")
            table.add_column("Kind")
            table.add_column("Tool")
            table.add_column("Content", overflow="fold")
            for item in page.items:
                table.add_row(
                    str(item.sequence),
                    item.item_kind,
                    item.tool_name or "-",
                    _preview(item.content),
                )
            console.print(table)
            emitted = True
            first = False
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    if not emitted:
        console.print(Panel("[dim]No conversation history.[/dim]", title="Conversation History"))


async def _emit_transcript_rows(
    history: RuntimeHistory,
    principal: Principal,
    execution_id: str,
) -> None:
    console = get_console()
    cursor: str | None = None
    emitted = False
    first = True
    while True:
        page = await history.transcript(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        if page.items:
            table = Table(
                title="Transcript" if first else None,
                box=None,
                show_header=first,
            )
            table.add_column("#", justify="right", style="dim")
            table.add_column("Text", overflow="fold")
            for item in page.items:
                table.add_row(str(item.sequence), _preview(item.text))
            console.print(table)
            emitted = True
            first = False
        if page.next_cursor is None:
            break
        cursor = page.next_cursor
    if not emitted:
        console.print(Panel("[dim]No transcript.[/dim]", title="Transcript"))


async def _emit_model_interactions(
    history: RuntimeHistory,
    principal: Principal,
    execution_id: str,
    first_page: Page[ModelInteractionItem],
) -> None:
    console = get_console()
    page = first_page
    first = True
    emitted = False
    while True:
        items = page.items
        if items:
            table = Table(
                title="Model Requests" if first else None,
                box=None,
                show_header=first,
            )
            table.add_column("Req", justify="right")
            table.add_column("Seg/Depth")
            table.add_column("Purpose")
            table.add_column("Status")
            table.add_column("Model")
            table.add_column("Duration", justify="right")
            table.add_column("Tokens", justify="right")
            for item in items:
                table.add_row(
                    str(item.request_sequence),
                    f"{item.segment_sequence}/{item.depth}",
                    item.purpose,
                    _status_text(item.status),
                    _model_label(item.model),
                    _format_duration_ns(item.duration_ns),
                    _usage_label(item.usage),
                )
            console.print(table)
            emitted = True
            first = False
        cursor = page.next_cursor
        if cursor is None:
            break
        page = await history.model_interactions(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
    if not emitted:
        console.print(Panel("[dim]No model requests.[/dim]", title="Model Requests"))


def _prompt_architecture_tree(interaction: ModelInteractionItem) -> Tree:
    request = interaction.request
    parameters = request.get("parameters")
    parameter_map = parameters if isinstance(parameters, Mapping) else {}

    standing = _sequence(request.get("instructions"))
    instruction_parts = tuple(
        item
        for item in _sequence(parameter_map.get("instruction_parts"))
        if isinstance(item, Mapping)
    )
    fixed = tuple(item for item in instruction_parts if not item.get("dynamic", False))
    dynamic = tuple(item for item in instruction_parts if item.get("dynamic", False))
    messages = tuple(
        item for item in _sequence(request.get("messages")) if isinstance(item, Mapping)
    )
    function_tools = _sequence(parameter_map.get("function_tools"))
    native_tools = _sequence(parameter_map.get("native_tools"))
    revealed = _sequence(parameter_map.get("revealed_tool_names"))
    deferred = _sequence(parameter_map.get("deferred_capability_ids"))

    root = Tree(
        Text.assemble(
            ("Prompt Architecture", "bold"),
            (
                f"  request #{interaction.request_sequence} · "
                f"{interaction.purpose} · {_model_label(interaction.model)}",
                "dim",
            ),
        )
    )
    root.add(
        _layer_text(
            "Standing system",
            _count_size_summary(standing),
        )
    )
    root.add(
        _layer_text(
            "Fixed instruction prefix (F0/F1)",
            _instruction_parts_summary(fixed),
        )
    )
    root.add(
        _layer_text(
            "Dynamic overlay (O)",
            _instruction_parts_summary(dynamic),
        )
    )
    root.add(
        _layer_text(
            "Conversation context",
            _message_summary(messages),
        )
    )
    root.add(
        _layer_text(
            "Tool contract",
            (
                f"{len(function_tools)} function · {len(native_tools)} native · "
                f"{len(revealed)} revealed · {len(deferred)} deferred capabilities"
            ),
        )
    )
    root.add(
        _layer_text(
            "Output contract",
            _output_summary(parameter_map),
        )
    )
    return root


def _layer_text(label: str, detail: str) -> Text:
    return Text.assemble((label, "bold"), ("  "), (detail, "dim"))


def _count_size_summary(values: Sequence[object]) -> str:
    return f"{len(values)} item(s) · ~{_content_chars(values):,} chars"


def _instruction_parts_summary(values: Sequence[Mapping[object, object]]) -> str:
    names: list[str] = []
    for item in values:
        name = item.get("name")
        if isinstance(name, str) and name and name not in names:
            names.append(name)
    source = ""
    if names:
        source = " · sources: " + ", ".join(names[:6])
        if len(names) > 6:
            source += ", …"
    return f"{len(values)} part(s) · ~{_content_chars(values):,} chars{source}"


def _message_summary(messages: Sequence[Mapping[object, object]]) -> str:
    parts = 0
    kinds: dict[str, int] = {}
    for message in messages:
        message_parts = message.get("parts")
        if not isinstance(message_parts, list):
            continue
        parts += len(message_parts)
        for part in message_parts:
            if not isinstance(part, Mapping):
                continue
            kind = (
                part.get("part_kind")
                or part.get("kind")
                or part.get("type")
                or "other"
            )
            key = str(kind)
            kinds[key] = kinds.get(key, 0) + 1
    detail = ", ".join(f"{name}={count}" for name, count in sorted(kinds.items()))
    suffix = "" if not detail else f" · {detail}"
    return f"{len(messages)} message(s) · {parts} part(s){suffix}"


def _output_summary(parameters: Mapping[object, object]) -> str:
    mode = parameters.get("output_mode")
    text = parameters.get("allow_text_output")
    image = parameters.get("allow_image_output")
    return f"mode={mode or '-'} · text={_yes_no(text)} · image={_yes_no(image)}"


def _yes_no(value: object) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "-"


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    if value is None:
        return ()
    return (value,)


def _content_chars(value: object) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return sum(_content_chars(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return sum(_content_chars(item) for item in value)
    return 0


def _preview(value: object, limit: int = _PREVIEW_LIMIT) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            text = str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _model_label(model: Mapping[str, object]) -> str:
    for key in ("model_name", "route_id", "name", "model"):
        value = model.get(key)
        if isinstance(value, str) and value:
            return value
    return "-"


def _usage_label(usage: UsageMetrics | None) -> str:
    if usage is None:
        return "-"
    return (
        f"{usage.input_tokens:,} in / {usage.output_tokens:,} out"
        if usage.cache_read_tokens == 0 and usage.cache_write_tokens == 0
        else (
            f"{usage.input_tokens:,} in / {usage.output_tokens:,} out · "
            f"{usage.cache_read_tokens:,} cr / {usage.cache_write_tokens:,} cw"
        )
    )


def _format_duration_ns(value: int) -> str:
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


def _status_text(status: str) -> Text:
    style = {
        "SUCCEEDED": "green",
        "FAILED": "red",
        "CANCELLED": "yellow",
        "STARTED": "cyan",
        "RUNNING": "cyan",
    }.get(status, "")
    return Text(status, style=style)


command = Command()
