#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure one controlled warm sync Session turn with actual driver statements.

Run from the repository: python -m tests.ai._measure_sync_session_io --help.
By default this uses SQLite. Set LINKTOOLS_AI_BENCHMARK_MYSQL_URL to a
pre-provisioned, disposable MySQL benchmark database to use asyncmy instead.
The observers perform explicit active/idle source reads without host transport.
Reported timing includes the local fake model and not real provider latency.
"""

import argparse
import asyncio
import json
import os
from collections import Counter, defaultdict
from contextlib import AsyncExitStack, aclosing, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from uuid import uuid4

from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionDeltaType, ExecutionStatus
from linktools.ai.persistence import Persistence, PersistenceStorage, StateTurnResources
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend, current_sql_observation, sql_observation_scope

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


async def measure(args: argparse.Namespace) -> dict:
    entered, finish = asyncio.Event(), asyncio.Event()
    paused = False
    hints = asyncio.Event()
    invalidations = 0

    async def model(messages, info):
        if paused:
            entered.set()
            await finish.wait()
        return ModelResponse(parts=[TextPart("answer")])

    def invalidate() -> None:
        nonlocal invalidations
        invalidations += 1
        hints.set()

    namespace = f"sync-turn-{uuid4().hex}" if args.mysql_url else "sync-turn"

    def features() -> RuntimeFeatures:
        feature = RuntimeFeatures(namespace)
        feature.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
        feature.agent("default", model="default", allow_tools=())
        return model_features(_Models(_UsageFunctionModel(model)), feature)

    with TemporaryDirectory() as directory:
        async with AsyncExitStack() as stack:
            cache = InMemoryCacheBackend(max_bytes=2**24)
            runtimes = []
            for _ in range(2):
                coordination = StateTurnResources(pending_cache=cache)
                if args.mysql_url:
                    engine = create_async_engine(args.mysql_url, hide_parameters=True)
                    stack.push_async_callback(engine.dispose)
                    storage = PersistenceStorage.sql(engine, cache=cache, coordination=coordination)
                else:
                    storage = PersistenceStorage.sqlite(Path(directory) / "state.db", cache=cache,
                                                       coordination=coordination)
                state = await stack.enter_async_context(Persistence.open(namespace, storage=storage))
                runtimes.append(await stack.enter_async_context(Runtime.open(persistence=state, features=features())))
            owner, reader = runtimes
            await owner.agents.get().create_session("session")
            backend = owner._execution_service.runtime_backend()
            warm = await owner.sessions.start("session", "warm", principal=owner.default_principal)
            warm_owner = backend._turn_owners[warm.execution_id]
            assert (await warm.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
            await warm_owner.closed.wait()

            totals = Counter()
            phases = defaultdict(Counter)
            tx_events = Counter()
            sql_duration = Counter()
            statement_start = {}
            first_output_ns = None

            def before_sql(_conn, _cursor, _statement, _parameters, context, _executemany):
                statement_start[id(context)] = perf_counter_ns()

            def sql(_conn, _cursor, statement, _parameters, context, _executemany):
                phase = "/".join(current_sql_observation()) or "unattributed"
                sql_duration[phase] += perf_counter_ns() - statement_start.pop(id(context))
                verb = statement.lstrip().split(maxsplit=1)[0].upper()
                if verb == "WITH" and getattr(getattr(context.compiled, "statement", None), "is_select", False):
                    verb = "SELECT"
                if verb in {"BEGIN", "SELECT", "INSERT", "UPDATE", "DELETE", "WITH"}:
                    totals[verb] += 1
                    phases[phase][verb] += 1
                else:
                    totals["OTHER"] += 1
                    phases[phase]["OTHER"] += 1

            def begin(_connection):
                tx_events["begin"] += 1

            def commit(_connection):
                tx_events["commit"] += 1

            def rollback(_connection):
                tx_events["rollback"] += 1

            paused = True
            listeners = (("before_cursor_execute", before_sql), ("after_cursor_execute", sql),
                         ("begin", begin), ("commit", commit), ("rollback", rollback))
            for name, callback in listeners:
                event.listen(Engine, name, callback)
            unsub = None
            publication = None
            send_ns = perf_counter_ns()
            terminal_ns = send_ns
            try:
                if args.preflight:
                    with sql_observation_scope("fixture.owner_preflight"):
                        state = await owner.sessions.wait_inactive(
                            "session", principal=owner.default_principal, timeout_seconds=8,
                        )
                        assert state.active_execution_id is None
                if args.producer == "invalidation":
                    unsub = owner.sessions.observe_changes("session", invalidate)
                send_ns = perf_counter_ns()
                execution = await owner.sessions.start("session", "question", principal=owner.default_principal)
                active_owner = backend._turn_owners[execution.execution_id]
                capture = (owner.sessions.capture_published_turn("session", execution.execution_id, principal=owner.default_principal)
                           if args.terminal == "publication" else nullcontext())
                with capture as read_publication:
                    await asyncio.wait_for(entered.wait(), 8)
                    await execution.inspect()
                    async with AsyncExitStack() as scopes:
                        events = await scopes.enter_async_context(aclosing(
                            execution.watch(include_descendants=False, include_content=True),
                        ))
                        if unsub is None:
                            state_watch = await scopes.enter_async_context(aclosing(
                                owner.sessions.watch("session", principal=owner.default_principal),
                            ))
                            assert (await asyncio.wait_for(anext(state_watch), 8)).active_execution_id == execution.execution_id
                        else:
                            await asyncio.wait_for(hints.wait(), 8)
                            hints.clear()
                        with sql_observation_scope("fixture.subscriber_busy"):
                            busy = await reader.history.inspect_session("session", principal=reader.default_principal)
                        assert busy.active_execution_id == execution.execution_id

                        async def drain() -> None:
                            nonlocal first_output_ns
                            with sql_observation_scope("fixture.execution_stream"):
                                async for event_item in events:
                                    if (first_output_ns is None
                                            and event_item.event.event_type == ExecutionDeltaType.ASSISTANT_TEXT_DELTA):
                                        first_output_ns = perf_counter_ns()

                        stream = asyncio.create_task(drain())
                        try:
                            await asyncio.sleep(0)
                            finish.set()
                            assert (await execution.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
                            terminal_ns = perf_counter_ns()
                            await active_owner.closed.wait()
                            if unsub is None:
                                assert (await asyncio.wait_for(anext(state_watch), 8)).active_execution_id is None
                            else:
                                await asyncio.wait_for(hints.wait(), 8)
                            with sql_observation_scope("fixture.subscriber_idle"):
                                idle = await reader.history.inspect_session("session", principal=reader.default_principal)
                            assert idle.active_execution_id is None
                            await asyncio.wait_for(stream, 8)
                        finally:
                            if not stream.done():
                                stream.cancel()
                                await asyncio.gather(stream, return_exceptions=True)

                    if read_publication is None:
                        detail = await reader.history.session_detail("session", limit=1, max_age=0,
                                                                     principal=reader.default_principal)
                        assert detail.timeline.items[-1].execution_id == execution.execution_id
                    else:
                        publication = await read_publication()
                        assert publication is not None
            finally:
                finish.set()
                for name, callback in reversed(listeners):
                    event.remove(Engine, name, callback)
                if unsub is not None:
                    unsub()
            if publication is not None:
                detail = await reader.history.session_detail("session", limit=1, max_age=0,
                                                             principal=reader.default_principal)
                assert detail.timeline.items == (publication.turn,)
                assert detail.session.revision == publication.session_revision
            return {"statements": dict(totals), "phases": dict(sorted(phases.items())),
                    "sqlalchemy_transaction_events": dict(tx_events),
                    "driver_statement_ms": {key: value / 1e6 for key, value in sorted(sql_duration.items())},
                    "terminal_ms": (terminal_ns - send_ns) / 1e6,
                    "first_text_event_ms": None if first_output_ns is None else (first_output_ns - send_ns) / 1e6,
                    "source_invalidations": invalidations,
                    "database": "mysql" if args.mysql_url else "sqlite"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", choices=("snapshot", "invalidation"), default="invalidation")
    parser.add_argument("--terminal", choices=("history", "publication"), default="history")
    parser.add_argument("--preflight", action="store_true", help="also measure upstream idle-owner preflight")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be positive")
    args.mysql_url = os.environ.get("LINKTOOLS_AI_BENCHMARK_MYSQL_URL")
    if args.mysql_url and not args.mysql_url.startswith("mysql+asyncmy://"):
        parser.error("LINKTOOLS_AI_BENCHMARK_MYSQL_URL must use mysql+asyncmy")
    for sample in range(args.samples):
        print(json.dumps({"sample": sample + 1, **asyncio.run(measure(args))}, sort_keys=True))


if __name__ == "__main__":
    main()
