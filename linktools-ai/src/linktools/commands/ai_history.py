#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai-history`: inspect persisted local Runtime execution history."""

import asyncio
from argparse import Namespace
from collections import deque
from typing import TYPE_CHECKING

from linktools.ai.core import Principal, service_principal
from linktools.ai.errors import AIError
from linktools.ai.runtime import (
    ExecutionHistoryItem,
    ExecutionTraceItem,
    ExecutionView,
    ListExecutionRequest,
    ModelInteractionItem,
    RuntimeHistory,
    TranscriptItem,
)
from linktools.cli import BaseCommand, CommandError

from ._ai_common import _json_dumps, _load_workspace, _local_runtime_state

if TYPE_CHECKING:
    from linktools.cli import CommandParser

_PAGE_LIMIT = 200
_DEFAULT_LIST_LIMIT = 20


class Command(BaseCommand):
    """Inspect local AI execution history."""

    @property
    def name(self) -> str:
        return "ai-history"

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("execution_id", nargs="?", help="execution id")
        parser.add_argument(
            "--json",
            action="store_true",
            help="emit JSON output",
        )

    def run(self, args: Namespace) -> int:
        workspace = _load_workspace()

        async def execute() -> int:
            state = _local_runtime_state(workspace)
            async with RuntimeHistory.open(workspace, state=state) as history:
                principal = service_principal(history.tenant_id, "ai-history")
                if args.execution_id is None:
                    executions = await _recent_executions(history, principal)
                    _emit_execution_list(executions, as_json=args.json)
                    return 0
                payload = await _execution_detail(
                    history,
                    principal,
                    args.execution_id,
                )
                _emit_execution_detail(payload, as_json=args.json)
                return 0

        try:
            return asyncio.run(execute())
        except (AIError, TypeError, ValueError) as error:
            raise CommandError(str(error)) from error


async def _recent_executions(
    history: RuntimeHistory,
    principal: Principal,
    *,
    limit: int = _DEFAULT_LIST_LIMIT,
) -> tuple[ExecutionView, ...]:
    recent: deque[ExecutionView] = deque(maxlen=limit)
    cursor: str | None = None
    while True:
        page = await history.list_executions(
            ListExecutionRequest(
                principal=principal,
                cursor=cursor,
                limit=_PAGE_LIMIT,
            )
        )
        recent.extend(page.items)
        if page.next_cursor is None:
            return tuple(reversed(recent))
        cursor = page.next_cursor


async def _execution_detail(
    history: RuntimeHistory,
    principal: Principal,
    execution_id: str,
) -> dict[str, object]:
    execution = await history.inspect_execution(execution_id, principal=principal)
    return {
        "execution": execution,
        "history": await _all_history(history, execution_id, principal),
        "trace": await _all_trace(history, execution_id, principal),
        "transcript": await _all_transcript(history, execution_id, principal),
        "model_interactions": await _all_model_interactions(
            history,
            execution_id,
            principal,
        ),
    }


async def _all_history(
    history: RuntimeHistory,
    execution_id: str,
    principal: Principal,
) -> tuple[ExecutionHistoryItem, ...]:
    items: list[ExecutionHistoryItem] = []
    cursor: str | None = None
    while True:
        page = await history.history(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        items.extend(page.items)
        if page.next_cursor is None:
            return tuple(items)
        cursor = page.next_cursor


async def _all_trace(
    history: RuntimeHistory,
    execution_id: str,
    principal: Principal,
) -> tuple[ExecutionTraceItem, ...]:
    items: list[ExecutionTraceItem] = []
    cursor: str | None = None
    while True:
        page = await history.trace(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        items.extend(page.items)
        if page.next_cursor is None:
            return tuple(items)
        cursor = page.next_cursor


async def _all_transcript(
    history: RuntimeHistory,
    execution_id: str,
    principal: Principal,
) -> tuple[TranscriptItem, ...]:
    items: list[TranscriptItem] = []
    cursor: str | None = None
    while True:
        page = await history.transcript(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        items.extend(page.items)
        if page.next_cursor is None:
            return tuple(items)
        cursor = page.next_cursor


async def _all_model_interactions(
    history: RuntimeHistory,
    execution_id: str,
    principal: Principal,
) -> tuple[ModelInteractionItem, ...]:
    items: list[ModelInteractionItem] = []
    cursor: str | None = None
    while True:
        page = await history.model_interactions(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        items.extend(page.items)
        if page.next_cursor is None:
            return tuple(items)
        cursor = page.next_cursor


def _emit_execution_list(
    executions: tuple[ExecutionView, ...],
    *,
    as_json: bool,
) -> None:
    if as_json:
        print(_json_dumps({"executions": executions}))
        return
    if not executions:
        print("No executions.")
        return
    print("EXECUTION\tSTATUS\tAGENT\tSESSION\tPARENT")
    for execution in executions:
        print(
            "\t".join(
                (
                    execution.execution_id,
                    execution.status.value,
                    execution.agent_id,
                    execution.session_id or "-",
                    execution.parent_execution_id or "-",
                )
            )
        )


def _emit_execution_detail(payload: dict[str, object], *, as_json: bool) -> None:
    if as_json:
        print(_json_dumps(payload))
        return
    sections = (
        ("Execution", payload["execution"]),
        ("History", payload["history"]),
        ("Trace", payload["trace"]),
        ("Transcript", payload["transcript"]),
        ("Model interactions", payload["model_interactions"]),
    )
    for index, (title, value) in enumerate(sections):
        if index:
            print()
        print(title)
        if isinstance(value, tuple) and not value:
            print("  (none)")
        else:
            print(_json_dumps(value, indent=2))


command = Command()
