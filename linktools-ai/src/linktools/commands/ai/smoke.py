#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`linktools ai smoke`: exercise the real ACP stdio subprocess."""

import asyncio
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand
from linktools.ai.acp.errors import AcpDependencyError, require_sdk

if TYPE_CHECKING:
    from argparse import Namespace
    from linktools.cli import CommandParser


class SmokeClient:
    def __init__(self, trace: list[dict[str, object]]) -> None:
        self.trace = trace

    async def session_update(self, session_id: str, update: object, **kwargs: object) -> None:
        self.trace.append(
            {
                "event": "session_update",
                "session_id": session_id,
                "update": update.model_dump(mode="json"),
            }
        )

    async def request_permission(
        self,
        session_id: str,
        tool_call: object,
        options: list[object],
        **kwargs: object,
    ) -> object:
        import acp.schema as schema

        return schema.RequestPermissionResponse(
            outcome=schema.AllowedOutcome(optionId=options[0].option_id)
        )

    def on_connect(self, connection: object) -> None:
        self.connection = connection


class Command(BaseCommand):
    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--project", type=Path, default=None)
        parser.add_argument("--prompt", required=True)
        parser.add_argument("--timeout", type=float, default=60)
        parser.add_argument("--approval", choices=("allow", "deny"), default="deny")
        parser.add_argument("--json", action="store_true")
        parser.add_argument("--trace-file", type=Path, default=None)

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [AcpDependencyError]

    def run(self, args: "Namespace") -> int:
        return asyncio.run(_run(args))


async def _run(args: "Namespace") -> int:
    require_sdk()
    import acp
    from linktools.ai.cli.project import find_project_root

    project = find_project_root(args.project)
    trace = []
    try:
        async with acp.spawn_agent_process(
            SmokeClient(trace),
            sys.executable,
            "-m",
            "linktools",
            "ai",
            "acp",
            "--project",
            str(project),
            cwd=str(project),
            use_unstable_protocol=True,
        ) as (connection, process):
            await asyncio.wait_for(_smoke(connection, str(project), args.prompt), args.timeout)
            await connection.close()
            if process.returncode is None:
                process.terminate()
    except asyncio.TimeoutError as exc:
        raise AcpDependencyError("ACP smoke timed out") from exc
    if args.trace_file:
        args.trace_file.write_text("\n".join(json.dumps(item, sort_keys=True) for item in trace) + "\n", encoding="utf-8")
    payload = {"ok": True, "updates": len(trace)}
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    else:
        print(f"ACP smoke passed ({len(trace)} updates)")
    return 0


async def _smoke(connection: object, project: str, prompt: str) -> None:
    response = await connection.initialize(protocol_version=1)
    session = await connection.new_session(cwd=project)
    import acp.schema as schema

    await connection.prompt(
        session.session_id,
        [schema.TextContentBlock(type="text", text=prompt)],
    )
    await connection.close_session(session.session_id)


command = Command()
