#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""``lt ai history`` business logic.

Three read-only views over persisted state, all through
:class:`RuntimeClient` (no direct storage access):

* ``lt ai history``                 -- list sessions (id, run count, last updated)
* ``lt ai history <session>``       -- list that session's turns (seq, run, status, input preview)
* ``lt ai history <session> -r N``  -- show turn N's recorded model messages
* ``lt ai history <session> --run <id>`` -- show a run's full message trace

The local client enumerates persisted sessions/runs by scanning the local
backend's filesystem; a remote/SQL store would surface the same views through
its own enumeration. ``--json`` emits each view as one JSON line per item."""

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..client import build_runtime_client

if TYPE_CHECKING:
    from ..client import RuntimeClient


def run_history(
    *,
    session: "str | None",
    run_id: "str | None",
    turn: "int | None",
    project: "Path | None",
    remote: "str | None",
    json_output: bool,
) -> int:
    return asyncio.run(
        _history_async(
            session=session,
            run_id=run_id,
            turn=turn,
            project=project,
            remote=remote,
            json_output=json_output,
        )
    )


async def _history_async(
    *,
    session: "str | None",
    run_id: "str | None",
    turn: "int | None",
    project: "Path | None",
    remote: "str | None",
    json_output: bool,
    client: "RuntimeClient | None" = None,
) -> int:
    if client is None:
        # History is read-only; no model is needed.
        client = build_runtime_client(project=project, remote=remote, with_model=False)

    if session is None:
        await _list_sessions(client, json_output=json_output)
        return 0

    if run_id is not None:
        await _show_run(client, session, run_id, json_output=json_output)
        return 0

    if turn is not None:
        await _show_turn(client, session, turn, json_output=json_output)
        return 0

    await _list_turns(client, session, json_output=json_output)
    return 0


async def _list_sessions(client: "RuntimeClient", *, json_output: bool) -> None:
    sessions = await client.list_sessions()
    if not sessions:
        print("(no sessions)")
        return
    if json_output:
        for record in sessions:
            print(json.dumps(_session_summary(record), default=str))
        return
    print(f"{'SESSION':<24} {'RUNS':>6}  {'UPDATED':<26}")
    for record in sessions:
        sid = str(getattr(record, "id", "?"))
        updated = _updated_str(record)
        runs = await _session_run_count(client, sid)
        print(f"{sid:<24} {runs:>6}  {updated:<26}")


async def _list_turns(
    client: "RuntimeClient", session_id: str, *, json_output: bool
) -> None:
    turns = await client.list_session_turns(session_id)
    if not turns:
        print(f"(no turns in session {session_id})")
        return
    if json_output:
        for turn in turns:
            print(json.dumps(_turn_summary(turn), default=str))
        return
    print(f"{'SEQ':>4}  {'RUN':<14} {'STATUS':<11} {'CAPTURE':<12} INPUT")
    for turn in turns:
        seq = getattr(turn, "sequence", "?")
        run = str(getattr(turn, "run_id", "?"))[:12]
        status = _status_value(turn)
        capture = str(getattr(turn, "capture_state", ""))
        preview = _preview(getattr(turn, "input", ""))
        print(f"{seq:>4}  {run:<14} {status:<11} {capture:<12} {preview}")


async def _show_turn(
    client: "RuntimeClient", session_id: str, sequence: int, *, json_output: bool
) -> None:
    messages = await client.get_session_messages(session_id)
    # Session messages are indexed by turn sequence (1-based, in order).
    turns = [t for t in await client.list_session_turns(session_id)]
    idx = next(
        (i for i, t in enumerate(turns) if getattr(t, "sequence", -1) == sequence), None
    )
    if idx is None or idx >= len(messages):
        print(f"(turn {sequence} not found in session {session_id})")
        return
    delta = messages[idx]
    _emit_messages(delta, json_output=json_output)


async def _show_run(
    client: "RuntimeClient", session_id: str, run_id: str, *, json_output: bool
) -> None:
    record = await client.get_run(run_id)
    if record is None:
        print(f"(run {run_id} not found)")
        return
    if json_output:
        print(json.dumps({"run": _run_summary(record)}, default=str))
    else:
        print(f"run {run_id}  session={session_id}  status={_status_value(record)}")
    detail = await _safe_detail(client, run_id)
    if detail is None:
        return
    for interaction in detail.interactions:
        _emit_messages(_interaction_messages(interaction), json_output=json_output)


async def _safe_detail(client: "RuntimeClient", run_id: str) -> Any:
    # Run detail via the public RuntimeClient.get_run_detail (returns None when
    # the run is not visible or the client has no detail to report).
    try:
        return await client.get_run_detail(run_id)
    except Exception:
        return None


def _interaction_messages(interaction: Any) -> tuple:
    """Flatten one model interaction's request+response into a message list.

    The detail view's response is ``{"parts": [...]}`` without a ``kind`` tag;
    tag it ``response`` so :func:`_emit_messages` renders it as the assistant."""
    request = getattr(interaction, "request", None) or {}
    response = getattr(interaction, "response", None) or {}
    out: "list[Any]" = []
    req_messages = request.get("messages") if isinstance(request, dict) else None
    if req_messages:
        out.extend(req_messages)
    if response:
        tagged: "dict[str, Any]" = (
            dict(response) if isinstance(response, dict) else {"parts": response}
        )
        tagged.setdefault("kind", "response")
        out.append(tagged)
    return tuple(out)


def _emit_messages(messages: Any, *, json_output: bool) -> None:
    if not messages:
        print("(no messages recorded)")
        return
    for message in messages:
        if json_output:
            print(json.dumps(message, default=str))
            continue
        kind = message.get("kind") if isinstance(message, dict) else None
        role = {"request": "user", "response": "assistant"}.get(kind, kind or "?")
        for part in message.get("parts", ()) if isinstance(message, dict) else ():
            ptype = part.get("type") if isinstance(part, dict) else None
            content = (
                part.get("content") or part.get("text") or ""
                if isinstance(part, dict)
                else str(part)
            )
            if not content:
                continue
            label = (
                f"[{role}/{ptype or 'text'}]"
                if ptype and ptype != "text"
                else f"[{role}]"
            )
            print(f"{label} {content}")


async def _session_run_count(client: "RuntimeClient", session_id: str) -> int:
    runs = await client.list_runs()
    return sum(1 for r in runs if getattr(r, "session_id", None) == session_id)


def _session_summary(record: Any) -> dict:
    return {
        "id": getattr(record, "id", None),
        "tenant_id": getattr(record, "tenant_id", None),
        "updated_at": _updated_str(record),
    }


def _turn_summary(turn: Any) -> dict:
    return {
        "sequence": getattr(turn, "sequence", None),
        "run_id": getattr(turn, "run_id", None),
        "status": _status_value(turn),
        "capture_state": str(getattr(turn, "capture_state", "")),
        "input": getattr(turn, "input", None),
    }


def _run_summary(record: Any) -> dict:
    return {
        "id": getattr(record, "id", None),
        "session_id": getattr(record, "session_id", None),
        "status": _status_value(record),
    }


def _status_value(obj: Any) -> str:
    status = getattr(obj, "status", None)
    value = getattr(status, "value", status)
    return str(value) if value is not None else "?"


def _updated_str(record: Any) -> str:
    for attr in ("updated_at", "completed_at", "created_at"):
        value = getattr(record, attr, None)
        if value is not None:
            return str(value)
    return "?"


def _preview(value: Any, limit: int = 60) -> str:
    text = value if isinstance(value, str) else str(value)
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"
