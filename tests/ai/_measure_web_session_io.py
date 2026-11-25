#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure a canonical turn through ai-web's ASGI POST and single SSE stream.

Run with python -m tests.ai._measure_web_session_io. SQLite and a gated
local model are used. Warm is the default; --cold starts with empty history
and --cold-reopen seeds history then reopens the same SQLite database.
--step-off omits Step persistence. Session creation, seeding and the first
observer snapshot are excluded; owned producer cleanup is included.
The synthetic exact-turn readback is counted separately by default.
With --publication it is used only afterwards to verify SSE content.
"""

import argparse
import asyncio
import json
from contextlib import AsyncExitStack, asynccontextmanager
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from httpx import AsyncClient
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.step_persistence import StepPersistence
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import current_sql_observation, sql_observation_scope
from linktools.ai.web import create_app

from ._runtime_test_helpers import _Models, model_features
from .test_web_console import _HEADERS, client


async def measure(
    directory: Path, *, trace: bool = False, publication: bool = False,
    cold: bool = False, cold_reopen: bool = False, step_off: bool = False,
) -> dict:
    release, disconnect = asyncio.Event(), asyncio.Event()
    frames: asyncio.Queue[dict] = asyncio.Queue()

    async def model(_messages, _info):
        await release.wait()
        yield "controlled answer"

    storage = PersistenceStorage.sqlite(directory / "web.db")
    counts = Counter()
    phases = defaultdict(Counter)
    transactions = Counter()
    transaction_phases = defaultdict(Counter)
    writers: set[object] = set()

    def observe(connection, _cursor, statement, _parameters, context, _many):
        verb = statement.lstrip().split(maxsplit=1)[0].upper()
        if verb == "WITH":
            compiled = getattr(context.compiled, "statement", None)
            verb = next((kind.upper() for kind in ("select", "insert", "update", "delete")
                         if getattr(compiled, "is_" + kind, False)), verb)
        counts[verb] += 1
        phases["/".join(current_sql_observation()) or "unattributed"][verb] += 1
        if verb in {"INSERT", "UPDATE", "DELETE", "REPLACE"}:
            writers.add(connection)

    def count_transaction(kind):
        transactions[kind] += 1
        transaction_phases["/".join(current_sql_observation()) or "unattributed"][kind] += 1

    def begin(_connection):
        count_transaction("begin")

    def commit(connection):
        count_transaction("commit")
        if connection in writers:
            count_transaction("write_commit")
            writers.discard(connection)

    def rollback(connection):
        count_transaction("rollback")
        writers.discard(connection)

    class WorkspacePolicy(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource):
            await super().authorize(principal, action, resource)

    @asynccontextmanager
    async def open_app():
        features = RuntimeFeatures("web-turn-io")
        features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
        if trace:
            features.capability(TraceCapability(), scope="runtime")
        features.agent("default", model="default", allow_tools=())
        features = model_features(_Models(FunctionModel(stream_function=model)), features)
        async with AsyncExitStack() as stack:
            state = await stack.enter_async_context(Persistence.open("web-turn-io", storage=storage))
            if not step_off:
                native = await state.create_step_store()
                stack.push_async_callback(native.close)
                features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
            runtime = await stack.enter_async_context(Runtime.open(
                persistence=state, features=features, authorization=WorkspacePolicy(),
            ))
            runner = runtime.sessions._turn_runner
            assert runner is not None
            yield create_app(runtime=runtime, local_workspace=True), runner

    async def drain_producer(runner):
        async with asyncio.timeout(8):
            await asyncio.gather(*tuple(runner._tasks))

    async def create_session(http):
        assert (await http.get("/api/config")).json()["turn_sessions"]
        created = await http.post("/api/sessions", headers=_HEADERS,
                                  json={"session_id": "session", "request_id": "create"})
        assert created.status_code == 201, created.text

    async def submit(http, key):
        with sql_observation_scope("probe.message_post"):
            response = await http.post("/api/session/messages?session_id=session", headers=_HEADERS,
                                       json={"request_id": key, "prompt": "question"})
        assert response.status_code == 202, response.text
        return response.json()["turn_id"]

    if cold_reopen:
        async with open_app() as (app, runner), client(app) as http:
            await create_session(http)
            release.set()
            seed = await submit(http, "warm")
            await drain_producer(runner)
            assert answer(await read_turn(http, seed)) == "controlled answer"
        # Runtime, its native Step adapter and Persistence have closed. The
        # next producer opens the same path without the previous hot state.
        release.clear()

    async with open_app() as (app, runner):
        async def feed():
            path = "/api/session/events"
            async def receive():
                await disconnect.wait()
                return {"type": "http.disconnect"}

            pending = b""

            async def send(message):
                nonlocal pending
                if message["type"] == "http.response.start":
                    assert message["status"] == 200
                pending += message.get("body", b"")
                while b"\n\n" in pending:
                    frame, pending = pending.split(b"\n\n", 1)
                    if b"event: session_error" in frame:
                        raise AssertionError(frame.decode())
                    for line in frame.splitlines():
                        if line.startswith(b"data: "):
                            frames.put_nowait(json.loads(line[6:]))

            await app({"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
                       "scheme": "http", "path": path, "raw_path": path.encode(),
                       "query_string": b"session_id=session&view=chat", "headers": [],
                       "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345)}, receive, send)

        async def next_frame(kind, turn_id=None):
            async with asyncio.timeout(8):
                while True:
                    payload = await frames.get()
                    assert payload["type"] not in ("gap", "error"), payload
                    if payload["type"] == kind and (turn_id is None or payload["turn_id"] == turn_id):
                        return payload

        async with client(app) as http:
            if not cold_reopen:
                await create_session(http)

            stream = asyncio.create_task(feed())
            listeners = (("before_cursor_execute", observe), ("begin", begin), ("commit", commit), ("rollback", rollback))
            try:
                assert (await next_frame("snapshot"))["session"]["active_turn_id"] is None
                if not cold and not cold_reopen:
                    release.set()
                    warm = await submit(http, "warm")
                    assert (await next_frame("terminal", warm))["state"] == "succeeded"
                    await drain_producer(runner)
                    release.clear()
                for name, callback in listeners:
                    event.listen(Engine, name, callback)
                try:
                    turn_id = await submit(http, "measure")
                    await next_frame("admitted", turn_id)
                    release.set()
                    await next_frame("delta", turn_id)
                    terminal = await next_frame("terminal", turn_id)
                    assert terminal["state"] == "succeeded" and terminal.get("error_code") is None
                    assert answer(terminal) == "controlled answer"
                    await drain_producer(runner)
                    if not publication:
                        assert answer(await read_turn(http, turn_id)) == answer(terminal)
                finally:
                    disconnect.set()
                    release.set()
                    try:
                        await asyncio.wait_for(stream, 8)
                    finally:
                        for name, callback in reversed(listeners):
                            event.remove(Engine, name, callback)
            finally:
                if not stream.done():
                    stream.cancel()
                    await asyncio.gather(stream, return_exceptions=True)
            if publication:
                assert answer(await read_turn(http, turn_id)) == answer(terminal) == "controlled answer"
    # This readback exercises the fallback endpoint, not an induced SSE gap.
    repair_statements = phases.pop("probe.terminal_get", Counter())
    repair_transactions = transaction_phases.pop("probe.terminal_get", Counter())
    return {"statements": dict(counts - repair_statements), "transactions": dict(transactions - repair_transactions),
            "phases": dict(phases), "transaction_phases": dict(transaction_phases),
            "database": "sqlite", "warm": not (cold or cold_reopen), "trace": trace, "step": not step_off,
            "history": "cold_reopen" if cold_reopen else "empty" if cold else "warm",
            "http": {"session_sse": 1, "message_post": 1, "terminal_get": 0},
            "synthetic_repair_get": None if publication else {
                "requests": 1, "statements": dict(repair_statements), "transactions": dict(repair_transactions),
            }}


def answer(turn: dict) -> str:
    output = turn.get("output")
    if isinstance(output, dict) and "text" in output:
        return output["text"]
    return next(item["content"] for item in reversed(turn["items"]) if item["item_kind"] == "assistant")


async def read_turn(http: AsyncClient, turn_id: str) -> dict:
    with sql_observation_scope("probe.terminal_get"):
        response = await http.get("/api/session", params={
            "session_id": "session", "turn_id": turn_id,
        })
    assert response.status_code == 200, response.text
    turn = response.json()
    assert turn["turn_id"] == turn_id and turn["conversation_committed"]
    return turn

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    temperature = parser.add_mutually_exclusive_group()
    temperature.add_argument("--cold", action="store_true")
    temperature.add_argument("--cold-reopen", action="store_true")
    parser.add_argument("--step-off", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    for sample in range(args.samples):
        with TemporaryDirectory() as directory:
            result = asyncio.run(measure(
                Path(directory), trace=args.trace, publication=args.publication,
                cold=args.cold, cold_reopen=args.cold_reopen, step_off=args.step_off,
            ))
            print(json.dumps({"sample": sample + 1, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
