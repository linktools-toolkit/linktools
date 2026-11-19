#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure one controlled warm sync Session turn with actual SQLite statements.

Run from the repository: python -m tests.ai._measure_sync_session_io --help.
The observer issues explicit active and idle source reads; no transport, real
model, tools, driver checkout timing or deployed-server latency is simulated.
"""

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from contextlib import AsyncExitStack, aclosing, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
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

    def features() -> RuntimeFeatures:
        feature = RuntimeFeatures("sync-turn")
        feature.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.sync()), scope="runtime")
        feature.agent("default", model="default", allow_tools=())
        return model_features(_Models(_UsageFunctionModel(model)), feature)

    with TemporaryDirectory() as directory:
        async with AsyncExitStack() as stack:
            cache = InMemoryCacheBackend(max_bytes=2**24)
            runtimes = []
            for _ in range(2):
                storage = PersistenceStorage.sqlite(Path(directory) / "state.db", cache=cache,
                    coordination=StateTurnResources(pending_cache=cache))
                state = await stack.enter_async_context(Persistence.open("sync-turn", storage=storage))
                runtimes.append(await stack.enter_async_context(Runtime.attach(persistence=state, features=features())))
            owner, reader = runtimes
            await owner.agents.get().create_session("session")
            backend = owner._execution_service.runtime_backend()
            warm = await owner.sessions.start("session", "warm", principal=owner.default_principal)
            warm_owner = backend._turn_owners[warm.execution_id]
            assert (await warm.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
            await warm_owner.closed.wait()

            totals = Counter()
            phases = defaultdict(Counter)

            def sql(_conn, _cursor, statement, _parameters, context, _executemany):
                verb = statement.lstrip().split(maxsplit=1)[0].upper()
                if verb == "WITH" and getattr(getattr(context.compiled, "statement", None), "is_select", False):
                    verb = "SELECT"
                if verb in {"BEGIN", "SELECT", "INSERT", "UPDATE", "DELETE", "WITH"}:
                    totals[verb] += 1
                    phases["/".join(current_sql_observation()) or "unattributed"][verb] += 1

            paused = True
            event.listen(Engine, "after_cursor_execute", sql)
            unsub = None
            publication = None
            try:
                if args.preflight:
                    with sql_observation_scope("fixture.owner_preflight"):
                        state = await owner.sessions.wait_inactive(
                            "session", principal=owner.default_principal, timeout_seconds=8,
                        )
                        assert state.active_execution_id is None
                if args.producer == "invalidation":
                    unsub = owner.sessions.observe_changes("session", invalidate)
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
                            with sql_observation_scope("fixture.execution_stream"):
                                async for _ in events:
                                    pass

                        stream = asyncio.create_task(drain())
                        try:
                            await asyncio.sleep(0)
                            finish.set()
                            assert (await execution.wait(timeout_seconds=8)).result.status is ExecutionStatus.SUCCEEDED
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
                event.remove(Engine, "after_cursor_execute", sql)
                if unsub is not None:
                    unsub()
            if publication is not None:
                detail = await reader.history.session_detail("session", limit=1, max_age=0,
                                                             principal=reader.default_principal)
                assert detail.timeline.items == (publication.turn,)
                assert detail.session.revision == publication.session_revision
            return {"statements": dict(totals), "phases": dict(sorted(phases.items())),
                    "source_invalidations": invalidations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", choices=("snapshot", "invalidation"), default="invalidation")
    parser.add_argument("--terminal", choices=("history", "publication"), default="history")
    parser.add_argument("--preflight", action="store_true", help="also measure upstream idle-owner preflight")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    for sample in range(args.samples):
        print(json.dumps({"sample": sample + 1, **asyncio.run(measure(args))}, sort_keys=True))


if __name__ == "__main__":
    main()
