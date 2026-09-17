#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai-history`: inspect persisted local Runtime execution history."""

import json
from argparse import Namespace
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

from linktools.ai.core import Principal, service_principal
from linktools.ai.runtime import ExecutionInfo, RuntimeHistory
from linktools.cli import BaseCommand

from ._ai_common import _load_workspace, _local_runtime_state, _run_async

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
            async with RuntimeHistory.open(
                workspace,
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
    print("Execution")
    print(
        _json_dumps(
            await history.inspect_execution(execution_id, principal=principal),
            indent=2,
        )
    )

    for title, read_page in (
        ("History", history.history),
        ("Transcript", history.transcript),
        ("Model interactions", history.model_interactions),
    ):
        print()
        print(title)
        cursor: str | None = None
        emitted = False
        while True:
            page = await read_page(
                execution_id,
                principal=principal,
                cursor=cursor,
                limit=_PAGE_LIMIT,
            )
            for item in page.items:
                emitted = True
                print(_json_dumps(item, indent=2))
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        if not emitted:
            print("  (none)")


def _json_dumps(value: object, *, indent: "int | None" = None) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        sort_keys=True,
        default=_json_default,
    )


def _json_default(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


command = Command()
