#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai-history`: inspect persisted local Runtime execution history."""

import asyncio
from argparse import Namespace
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from linktools.ai.core import Principal, service_principal
from linktools.ai.errors import AIError
from linktools.ai.runtime import (
    ExecutionHistoryItem,
    ExecutionInfo,
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

    def run(self, args: Namespace) -> int:
        workspace = _load_workspace()

        async def execute() -> int:
            state = _local_runtime_state(workspace)
            async with RuntimeHistory.open(workspace, state=state) as history:
                principal = service_principal(history.tenant_id, "ai-history")
                if args.execution_id is None:
                    executions = await _recent_executions(history, principal)
                    _emit_execution_list(executions)
                    return 0
                await _emit_execution_detail(
                    history,
                    principal,
                    args.execution_id,
                )
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
) -> tuple[ExecutionInfo, ...]:
    return await history.recent_executions(principal=principal, limit=limit)


async def _history_items(
    history: RuntimeHistory,
    execution_id: str,
    principal: Principal,
) -> AsyncIterator[ExecutionHistoryItem]:
    cursor: str | None = None
    while True:
        page = await history.history(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        for item in page.items:
            yield item
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


async def _transcript_items(
    history: RuntimeHistory,
    execution_id: str,
    principal: Principal,
) -> AsyncIterator[TranscriptItem]:
    cursor: str | None = None
    while True:
        page = await history.transcript(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        for item in page.items:
            yield item
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


async def _model_interaction_items(
    history: RuntimeHistory,
    execution_id: str,
    principal: Principal,
) -> AsyncIterator[ModelInteractionItem]:
    cursor: str | None = None
    while True:
        page = await history.model_interactions(
            execution_id,
            principal=principal,
            cursor=cursor,
            limit=_PAGE_LIMIT,
        )
        for item in page.items:
            yield item
        if page.next_cursor is None:
            return
        cursor = page.next_cursor


def _emit_execution_list(executions: tuple[ExecutionInfo, ...]) -> None:
    if not executions:
        print("No executions.")
        return
    print("CREATED\tEXECUTION\tSTATUS\tAGENT\tSESSION\tPARENT\tERROR")
    for execution in executions:
        print(
            "\t".join(
                (
                    execution.created_at.isoformat(),
                    execution.execution_id,
                    execution.status.value,
                    execution.agent_id,
                    execution.session_id or "-",
                    execution.parent_execution_id or "-",
                    execution.error_code or "-",
                )
            )
        )


async def _emit_execution_detail(
    history: RuntimeHistory,
    principal: Principal,
    execution_id: str,
) -> None:
    execution = await history.inspect_execution(execution_id, principal=principal)
    print("Execution")
    print(_json_dumps(execution, indent=2))

    await _emit_items(
        "History",
        _history_items(history, execution_id, principal),
    )
    await _emit_items(
        "Transcript",
        _transcript_items(history, execution_id, principal),
    )
    await _emit_items(
        "Model interactions",
        _model_interaction_items(history, execution_id, principal),
    )


async def _emit_items(title: str, items: AsyncIterator[object]) -> None:
    print()
    print(title)
    emitted = False
    async for item in items:
        emitted = True
        print(_json_dumps(item, indent=2))
    if not emitted:
        print("  (none)")


command = Command()
