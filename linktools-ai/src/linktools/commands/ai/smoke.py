#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""`linktools ai smoke`: verify the real ACP stdio subprocess."""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from linktools.cli import BaseCommand
from linktools.ai.acp.errors import AcpDependencyError, require_sdk

if TYPE_CHECKING:
    from argparse import Namespace
    from linktools.cli import CommandParser


@dataclass(slots=True)
class SmokeResult:
    ok: bool = False
    protocol_version: int = 1
    session_id: "str | None" = None
    stop_reason: "str | None" = None
    update_count: int = 0
    message_chunk_count: int = 0
    tool_call_count: int = 0
    permission_request_count: int = 0
    process_exit_code: "int | None" = None
    elapsed_ms: int = 0
    error: "str | None" = None
    update_times: "list[float]" = field(default_factory=list)
    prompt_response_time: "float | None" = None

    def as_json(self) -> "dict[str, object]":
        return {
            "ok": self.ok,
            "protocol_version": self.protocol_version,
            "session_id": self.session_id,
            "stop_reason": self.stop_reason,
            "update_count": self.update_count,
            "message_chunk_count": self.message_chunk_count,
            "tool_call_count": self.tool_call_count,
            "permission_request_count": self.permission_request_count,
            "process_exit_code": self.process_exit_code,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


class SmokeClient:
    def __init__(self, trace: "list[dict[str, object]]", approval_policy: "Literal['allow', 'deny']") -> None:
        self.trace = trace
        self.approval_policy = approval_policy
        self.permission_request_count = 0
        self.update_times: "list[float]" = []
        self.selected_permission_kinds: "list[str]" = []

    async def session_update(self, session_id: str, update: object, **kwargs: object) -> None:
        now = time.monotonic()
        self.update_times.append(now)
        payload = update.model_dump(mode="json", by_alias=True)
        self.trace.append({"event": "session_update", "session_id": session_id, "update": payload})

    async def request_permission(
        self,
        session_id: str,
        tool_call: object,
        options: "list[object]",
        **kwargs: object,
    ) -> object:
        import acp.schema as schema

        self.permission_request_count += 1
        required_kind = "allow_once" if self.approval_policy == "allow" else "reject_once"
        selected = next((option for option in options if option.kind == required_kind), None)
        if selected is None:
            raise RuntimeError(f"permission option {required_kind!r} was not advertised")
        self.selected_permission_kinds.append(selected.kind)
        return schema.RequestPermissionResponse(
            outcome=schema.AllowedOutcome(
                optionId=selected.option_id,
                outcome="selected",
            )
        )

    def on_connect(self, connection: object) -> None:
        self.connection = connection


class SmokeAssertionError(RuntimeError):
    pass


class Command(BaseCommand):
    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--project", type=Path, default=None)
        parser.add_argument("--prompt", required=True)
        parser.add_argument("--timeout", type=float, default=60)
        parser.add_argument("--approval", choices=("allow", "deny"), default="deny")
        parser.add_argument("--expected-stop-reason", choices=("end_turn", "cancelled"), default="end_turn")
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
    trace: "list[dict[str, object]]" = []
    client = SmokeClient(trace, args.approval)
    result = SmokeResult()
    started = time.monotonic()
    process = None
    stderr_task: "asyncio.Task[bytes] | None" = None
    failure_code = 0
    try:
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(project), child_env.get("PYTHONPATH", "")) if item
        )
        async with acp.spawn_agent_process(
            client,
            sys.executable,
            "-m",
            "linktools",
            "ai",
            "acp",
            "--project",
            str(project),
            cwd=str(project),
            env=child_env,
            use_unstable_protocol=True,
        ) as (connection, process):
            if process.stderr is not None:
                stderr_task = asyncio.create_task(process.stderr.read())
            result = await asyncio.wait_for(
                _smoke(connection, str(project), args.prompt),
                args.timeout,
            )
    except asyncio.TimeoutError as exc:
        result.error = "smoke timed out"
        failure_code = 5
    except SmokeAssertionError as exc:
        result.error = str(exc)
        failure_code = 4
    except Exception as exc:
        result.error = f"ACP subprocess or transport failed: {type(exc).__name__}"
        failure_code = 3
    finally:
        if stderr_task is not None:
            await asyncio.gather(stderr_task, return_exceptions=True)
        result.process_exit_code = process.returncode if process is not None else None
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        result.update_count = len(trace)
        result.message_chunk_count = sum(
            item.get("update", {}).get("sessionUpdate") == "agent_message_chunk"
            for item in trace
        )
        result.tool_call_count = sum(
            item.get("update", {}).get("sessionUpdate") == "tool_call_update"
            and item.get("update", {}).get("status") == "completed"
            for item in trace
        )
        result.permission_request_count = client.permission_request_count
        result.update_times = client.update_times
        if result.process_exit_code != 0 and failure_code == 0:
            result.error = "ACP subprocess exited unsuccessfully"
            failure_code = 3
        if failure_code == 0:
            if result.update_count == 0:
                result.error = "ACP prompt produced zero updates"
                failure_code = 4
            elif result.message_chunk_count == 0 and result.stop_reason != "cancelled":
                result.error = "ACP prompt produced zero agent message chunks"
                failure_code = 4
            elif result.prompt_response_time is None:
                result.error = "ACP prompt produced no response"
                failure_code = 4
            elif result.stop_reason != (
                "cancelled"
                if (
                    args.approval == "deny"
                    and result.permission_request_count > 0
                    and args.expected_stop_reason == "end_turn"
                )
                else args.expected_stop_reason
            ):
                result.error = f"unexpected stop reason: {result.stop_reason}"
                failure_code = 4
            elif result.tool_call_count > 0 and result.permission_request_count == 0:
                result.error = "tool call did not trigger a permission request"
                failure_code = 4
            elif result.permission_request_count > 0 and not client.selected_permission_kinds:
                result.error = "permission policy was not applied"
                failure_code = 4
            elif any(timestamp >= result.prompt_response_time for timestamp in result.update_times):
                result.error = "ACP update arrived after PromptResponse"
                failure_code = 4
            else:
                result.ok = True
        if args.trace_file:
            args.trace_file.write_text(
                "\n".join(json.dumps(item, sort_keys=True) for item in trace) + "\n",
                encoding="utf-8",
            )
        _print_result(args, result)
    return failure_code if failure_code else 0


async def _smoke(connection: object, project: str, prompt: str) -> SmokeResult:
    import acp.schema as schema

    result = SmokeResult()
    initialized = await connection.initialize(protocol_version=1)
    result.protocol_version = initialized.protocol_version
    if result.protocol_version != 1:
        raise SmokeAssertionError("negotiated protocol version is not 1")
    session = await connection.new_session(cwd=project)
    result.session_id = session.session_id
    try:
        response = await connection.prompt(
            session.session_id,
            [schema.TextContentBlock(type="text", text=prompt)],
        )
    except Exception as exc:
        raise SmokeAssertionError("ACP prompt failed") from exc
    result.prompt_response_time = time.monotonic()
    result.stop_reason = response.stop_reason
    if result.stop_reason not in {"end_turn", "cancelled"}:
        raise SmokeAssertionError(f"unexpected stop reason: {result.stop_reason}")
    try:
        await connection.close_session(session.session_id)
    except Exception as exc:
        raise SmokeAssertionError("ACP session close failed") from exc
    return result


def _print_result(args: "Namespace", result: SmokeResult) -> None:
    payload = result.as_json()
    if args.json:
        print(json.dumps(payload, sort_keys=True))
    elif result.ok:
        print(f"ACP smoke passed ({result.update_count} updates)")
    else:
        print(f"ACP smoke failed: {result.error or 'unknown failure'}", file=sys.stderr)


command = Command()
