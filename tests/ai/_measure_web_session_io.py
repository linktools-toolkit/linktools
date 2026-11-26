#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure a canonical turn through ai-web's ASGI POST and SSE streams.

Run with python -m tests.ai._measure_web_session_io. SQLite and a gated
local model are used by default. --mysql uses LINKTOOLS_AI_BENCHMARK_MYSQL_URL
and requires a pre-provisioned, disposable MySQL database; this probe never
creates, alters, or drops MySQL tables. --tabs accepts one or two subscribers.
Warm is the default; --cold starts with empty history and --cold-reopen seeds
history then reopens the same database. --step-off omits Step persistence.
Session creation, seeding and initial SSE snapshots are excluded. Healthy
publication is measured without a terminal GET; --measure-repair opts into a
synthetic exact-turn GET reported separately (not an induced gap).

The clients call the real ASGI HTTP app in-process; this is not a browser
benchmark or a comparison with an older revision.
"""

import argparse
import asyncio
import json
import os
from contextlib import AsyncExitStack, asynccontextmanager
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from httpx import AsyncClient
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.step_persistence import StepPersistence
from sqlalchemy import event
from sqlalchemy.engine import Engine, URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import current_sql_observation, sql_observation_scope
from linktools.ai.web import create_app

from ._runtime_test_helpers import _Models, model_features
from .test_web_console import _HEADERS, client


async def measure(
    directory: Path, *, trace: bool = False, publication: bool = True,
    cold: bool = False, cold_reopen: bool = False, step_off: bool = False,
    tabs: int = 1, mysql_url: URL | None = None,
) -> dict:
    async with AsyncExitStack() as stack:
        engine = None if mysql_url is None else create_async_engine(mysql_url, hide_parameters=True)
        if engine is not None:
            stack.push_async_callback(engine.dispose)
        storage = (PersistenceStorage.sqlite(directory / "web.db") if engine is None
                   else PersistenceStorage.sql(engine))
        namespace = "web-turn-io" if engine is None else f"web-turn-io-{uuid4().hex}"
        return await _measure(
            storage, namespace, engine=None if engine is None else engine.sync_engine,
            trace=trace, publication=publication, cold=cold, cold_reopen=cold_reopen,
            step_off=step_off, tabs=tabs,
        )


async def _measure(
    storage: PersistenceStorage, namespace: str, *, engine: Engine | None,
    trace: bool, publication: bool, cold: bool, cold_reopen: bool, step_off: bool, tabs: int,
) -> dict:
    release, disconnect = asyncio.Event(), asyncio.Event()
    frames = tuple(asyncio.Queue() for _ in range(tabs))

    async def model(_messages, _info):
        await release.wait()
        yield "controlled answer"

    counts = Counter()
    phases = defaultdict(Counter)
    transactions = Counter()
    transaction_phases = defaultdict(Counter)
    writers: set[object] = set()
    http_counts = Counter()
    measure_requests = False

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
        features = RuntimeFeatures(namespace)
        features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
        if trace:
            features.capability(TraceCapability(), scope="runtime")
        features.agent("default", model="default", allow_tools=())
        features = model_features(_Models(FunctionModel(stream_function=model)), features)
        async with AsyncExitStack() as stack:
            state = await stack.enter_async_context(Persistence.open(namespace, storage=storage))
            if not step_off:
                native = await state.create_step_store()
                stack.push_async_callback(native.close)
                features.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
            runtime = await stack.enter_async_context(Runtime.open(
                persistence=state, features=features, authorization=WorkspacePolicy(),
            ))
            runner = runtime.sessions._turn_runner
            assert runner is not None
            app = create_app(runtime=runtime, local_workspace=True)

            async def observed_app(scope, receive, send):
                if scope["type"] == "http":
                    endpoint = scope["method"], scope["path"]
                    if endpoint == ("GET", "/api/session/events"):
                        http_counts["session_sse"] += 1
                    elif measure_requests:
                        if endpoint == ("POST", "/api/session/messages"):
                            http_counts["message_post"] += 1
                        elif endpoint == ("GET", "/api/session") and b"turn_id=" in scope["query_string"]:
                            http_counts["terminal_get"] += 1
                await app(scope, receive, send)

            yield observed_app, runner

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
        async def feed(queue: asyncio.Queue):
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
                            queue.put_nowait(json.loads(line[6:]))

            await app({"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
                       "scheme": "http", "path": path, "raw_path": path.encode(),
                       "query_string": b"session_id=session&view=chat", "headers": [],
                       "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345)}, receive, send)

        async def next_frames(kind, turn_id=None):
            async def next_frame(queue):
                while True:
                    payload = await queue.get()
                    assert payload["type"] not in ("gap", "error"), payload
                    if payload["type"] == kind and (turn_id is None or payload["turn_id"] == turn_id):
                        return payload

            async with asyncio.timeout(8):
                return await asyncio.gather(*(next_frame(queue) for queue in frames))

        async with client(app) as http:
            if not cold_reopen:
                await create_session(http)

            # Observe each connection at the real app before counting SQL.
            streams = tuple(asyncio.create_task(feed(queue)) for queue in frames)
            observation_target = engine or Engine
            listeners = (("before_cursor_execute", observe), ("begin", begin), ("commit", commit), ("rollback", rollback))
            try:
                assert all(frame["session"]["active_turn_id"] is None for frame in await next_frames("snapshot"))
                if not cold and not cold_reopen:
                    release.set()
                    warm = await submit(http, "warm")
                    assert all(frame["state"] == "succeeded" for frame in await next_frames("terminal", warm))
                    await drain_producer(runner)
                    release.clear()
                for name, callback in listeners:
                    event.listen(observation_target, name, callback)
                measure_requests = True
                started = perf_counter()
                try:
                    turn_id = await submit(http, "measure")
                    await next_frames("admitted", turn_id)
                    release.set()
                    await next_frames("delta", turn_id)
                    terminals = await next_frames("terminal", turn_id)
                    terminal_seconds = perf_counter() - started
                    for terminal in terminals:
                        assert terminal["state"] == "succeeded" and terminal.get("error_code") is None
                        assert answer(terminal) == "controlled answer"
                    await drain_producer(runner)
                    drained_seconds = perf_counter() - started
                    if not publication:
                        assert answer(await read_turn(http, turn_id)) == answer(terminals[0])
                finally:
                    measure_requests = False
                    for name, callback in reversed(listeners):
                        event.remove(observation_target, name, callback)
                    disconnect.set()
                    release.set()
                    await asyncio.wait_for(asyncio.gather(*streams), 8)
            finally:
                for stream in streams:
                    if not stream.done():
                        stream.cancel()
                await asyncio.gather(*streams, return_exceptions=True)
    assert http_counts == {"session_sse": tabs, "message_post": 1,
                           **({} if publication else {"terminal_get": 1})}, http_counts
    # This readback exercises the fallback endpoint, not an induced SSE gap.
    repair_statements = phases.pop("probe.terminal_get", Counter())
    repair_transactions = transaction_phases.pop("probe.terminal_get", Counter())
    return {"statements": dict(counts - repair_statements), "transactions": dict(transactions - repair_transactions),
            "phases": dict(phases), "transaction_phases": dict(transaction_phases),
            "database": "sqlite" if engine is None else "mysql", "warm": not (cold or cold_reopen), "trace": trace, "step": not step_off,
            "history": "cold_reopen" if cold_reopen else "empty" if cold else "warm",
            "transport": "in_process_asgi",
            "elapsed_seconds": {"all_terminal_frames": terminal_seconds, "producer_drained": drained_seconds},
            "http": {"session_sse": http_counts["session_sse"], "message_post": http_counts["message_post"],
                     "terminal_get": 0},
            "synthetic_repair_get": None if publication else {
                "requests": http_counts["terminal_get"], "statements": dict(repair_statements),
                "transactions": dict(repair_transactions),
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
    parser.add_argument("--mysql", action="store_true", help="use a pre-provisioned isolated MySQL database")
    parser.add_argument("--tabs", type=int, choices=(1, 2), default=1)
    publication = parser.add_mutually_exclusive_group()
    publication.add_argument("--publication", action="store_true", default=True)
    publication.add_argument("--measure-repair", action="store_false", dest="publication")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    mysql_url = None
    if args.mysql:
        raw_url = os.environ.get("LINKTOOLS_AI_BENCHMARK_MYSQL_URL")
        try:
            mysql_url = make_url(raw_url) if raw_url else None
        except ArgumentError:
            parser.error("LINKTOOLS_AI_BENCHMARK_MYSQL_URL must be a valid isolated MySQL URL")
        if mysql_url is None or mysql_url.drivername not in {"mysql+asyncmy", "mysql+aiomysql"} or not mysql_url.database:
            parser.error("LINKTOOLS_AI_BENCHMARK_MYSQL_URL must specify a pre-provisioned MySQL database and async driver")
    for sample in range(args.samples):
        with TemporaryDirectory() as directory:
            result = asyncio.run(measure(
                Path(directory), trace=args.trace, publication=args.publication,
                cold=args.cold, cold_reopen=args.cold_reopen, step_off=args.step_off,
                tabs=args.tabs, mysql_url=mysql_url,
            ))
            print(json.dumps({"sample": sample + 1, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
