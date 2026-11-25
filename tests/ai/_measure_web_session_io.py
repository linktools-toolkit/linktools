#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure a warm retained turn through ai-web's actual ASGI endpoints.

Run with python -m tests.ai._measure_web_session_io. SQLite and a gated
local model are used. The initial session and initial observer reads are
excluded.  Owned cleanup is included. By default the exact-turn readback is
included. With --publication it is only used afterwards to verify SSE content.
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from httpx import AsyncClient
from pydantic_ai.models.function import FunctionModel
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.capability import ExecutionCapability, StepPersistenceCapability, TraceCapability
from linktools.ai.core import CommitPolicy
from linktools.ai.persistence import PersistenceStorage, StateTurnResources
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import current_sql_observation
from linktools.ai.web import create_app

from ._runtime_test_helpers import _Models, model_features
from .test_web_console import _HEADERS, client


async def measure(directory: Path, *, trace: bool = False, publication: bool = False) -> dict:
    release, disconnect = asyncio.Event(), asyncio.Event()
    frames: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
    committed: dict | None = None

    async def model(_messages, _info):
        await release.wait()
        yield "controlled answer"

    features = RuntimeFeatures("web-turn-io")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()))
    features.capability(StepPersistenceCapability())
    if trace:
        features.capability(TraceCapability())
    features.agent("default", model="default", allow_tools=())
    features = model_features(_Models(FunctionModel(stream_function=model)), features)
    storage = PersistenceStorage.sqlite(directory / "web.db", coordination=StateTurnResources())
    counts = Counter()
    phases = defaultdict(Counter)
    transactions = Counter()
    writers: set[object] = set()

    def observe(connection, _cursor, statement, _parameters, context, _many):
        verb = statement.lstrip().split(maxsplit=1)[0].upper()
        if verb == "WITH" and getattr(getattr(context.compiled, "statement", None), "is_select", False):
            verb = "SELECT"
        counts[verb] += 1
        phases["/".join(current_sql_observation()) or "unattributed"][verb] += 1
        if verb in {"INSERT", "UPDATE", "DELETE", "REPLACE"}:
            writers.add(connection)

    def begin(_connection):
        transactions["begin"] += 1

    def commit(connection):
        transactions["commit"] += 1
        if connection in writers:
            transactions["write_commit"] += 1
            writers.discard(connection)

    def rollback(connection):
        transactions["rollback"] += 1
        writers.discard(connection)

    async with Runtime.open("web-turn-io", storage=storage, features=features) as runtime:
        app = create_app(runtime=runtime)

        async def feed():
            path = "/api/session/events"
            async def receive():
                await disconnect.wait()
                return {"type": "http.disconnect"}

            async def send(message):
                if message["type"] == "http.response.start":
                    assert message["status"] == 200
                kind = "message"
                for line in message.get("body", b"").decode().splitlines():
                    if line.startswith("event: "):
                        kind = line[7:]
                    elif line.startswith("data: "):
                        frames.put_nowait((kind, json.loads(line[6:])))

            await app({"type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
                       "scheme": "http", "path": path, "raw_path": path.encode(),
                       "query_string": b"session_id=session&view=chat", "headers": [],
                       "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345)}, receive, send)

        async def next_occupancy():
            nonlocal committed
            while True:
                kind, payload = await asyncio.wait_for(frames.get(), 8)
                assert kind != "session_error", payload
                if kind == "session_turn":
                    committed = payload["turn"]
                elif kind == "message":
                    return payload

        async with client(app) as http:
            created = await http.post("/api/sessions", headers=_HEADERS,
                                      json={"session_id": "session", "request_id": "create"})
            assert created.status_code == 201, created.text
            release.set()
            # Warm the same API and worker path, not just the connection pool.
            async def submit(key):
                response = await http.post("/api/session/messages?session_id=session", headers=_HEADERS,
                                           json={"request_id": key, "prompt": "question"})
                assert response.status_code == 202, response.text
                return response.json()["execution_id"]

            warm = await submit("warm")
            async with asyncio.timeout(8):
                await (await runtime.executions.get(warm, principal=runtime.default_principal)).wait()
                owner = runtime._execution_service.runtime_backend()._turn_owners.get(warm)
                if owner is not None:
                    await owner.closed.wait()
            release.clear()
            stream = asyncio.create_task(feed())
            listeners = (("before_cursor_execute", observe), ("begin", begin), ("commit", commit), ("rollback", rollback))
            try:
                assert (await next_occupancy())["active_execution_id"] is None
                for name, callback in listeners:
                    event.listen(Engine, name, callback)
                try:
                    execution_id = await submit("measure")
                    assert (await next_occupancy())["active_execution_id"] == execution_id
                    owner = runtime._execution_service.runtime_backend()._turn_owners[execution_id]
                    release.set()
                    assert (await next_occupancy())["active_execution_id"] is None
                    if publication:
                        assert committed is not None, "no committed turn on the live stream"
                        turn = committed
                    else:
                        turn = await read_turn(http, execution_id)
                    assert turn["execution_id"] == execution_id and turn["conversation_committed"]
                    await asyncio.wait_for(owner.closed.wait(), 8)
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
                assert committed == await read_turn(http, execution_id)
    return {"statements": dict(counts), "transactions": dict(transactions),
            "phases": dict(phases), "trace": trace,
            "http": {"session_sse": 1, "message_post": 1, "terminal_get": int(not publication)}}


async def read_turn(http: AsyncClient, execution_id: str) -> dict:
    response = await http.get("/api/session", params={
        "session_id": "session", "execution_id": execution_id,
    })
    assert response.status_code == 200, response.text
    turn, = response.json()["timeline"]["items"]
    return turn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--publication", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    for sample in range(args.samples):
        with TemporaryDirectory() as directory:
            result = asyncio.run(measure(Path(directory), trace=args.trace, publication=args.publication))
            print(json.dumps({"sample": sample + 1, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
