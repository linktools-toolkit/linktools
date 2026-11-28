#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A configured split Step route retains the canonical conversation transcript."""

import asyncio
from contextlib import AsyncExitStack
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import SessionCapability
from linktools.ai.core import Principal, TenantAuthorizationPolicy
from linktools.ai.history import History
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import (
    Persistence, PersistenceDomain, PersistencePlan, PersistenceStorage, StateTurnResources,
)
from linktools.ai.persistence.turns import SessionTurnKey
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageRoute

from ._metrics_server_test_helpers import _server_database
from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _SessionPolicy(TenantAuthorizationPolicy):
    """Explicit host policy for this fixture's session."""


@pytest.mark.asyncio
async def test_split_step_text_turn_reopens_through_canonical_history(tmp_path: Path) -> None:
    conversation = StorageRoute.sqlite(tmp_path / "conversation.db")
    step = StorageRoute.sqlite(tmp_path / "step.db")
    storage = PersistenceStorage(PersistencePlan(conversation=conversation, execution=step, checkpoint=step))
    await _assert_split_step_text_roundtrip(storage)


@pytest.mark.asyncio
async def test_mysql_split_step_text_turn_reopens_through_canonical_history(_mysql_server) -> None:
    async with AsyncExitStack() as stack:
        routes = []
        for _ in range(2):
            _, command = stack.enter_context(_server_database(_mysql_server))
            socket = next(option.split("=", 1)[1] for option in command if option.startswith("--socket="))
            url = URL.create("mysql+asyncmy", username="root", database=command[-1], query={"unix_socket": socket})
            engine = create_async_engine(url, isolation_level="REPEATABLE READ", hide_parameters=True)
            stack.push_async_callback(engine.dispose)
            await provision_database(engine)
            # Establish each physical database's archive identity using the
            # existing public owner initialization, before making a split plan.
            async with Persistence.open("split-text", storage=PersistenceStorage.sql(engine)) as state:
                native = await state.create_step_store()
                await native.close()
            routes.append(StorageRoute.sql(engine))
        conversation, step = routes
        storage = PersistenceStorage(
            PersistencePlan(conversation=conversation, execution=step, checkpoint=step),
            coordination=StateTurnResources(),
        )
        await _assert_split_step_text_roundtrip(storage)


@pytest.mark.asyncio
@pytest.mark.parametrize("initialized", (True, False), ids=("same-archive", "missing-archive"))
async def test_mysql_split_rejects_alias_without_provisioning(_mysql_server, initialized: bool) -> None:
    async with AsyncExitStack() as stack:
        _, command = stack.enter_context(_server_database(_mysql_server))
        socket = next(option.split("=", 1)[1] for option in command if option.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database=command[-1], query={"unix_socket": socket})
        source, alias = (create_async_engine(url, hide_parameters=True) for _ in range(2))
        for engine in (source, alias):
            stack.push_async_callback(engine.dispose)
        await provision_database(source)
        if initialized:
            async with Persistence.open("split-alias", storage=PersistenceStorage.sql(source)) as state:
                await (await state.create_step_store()).close()

        step = StorageRoute.sql(source)
        storage = PersistenceStorage(
            PersistencePlan(conversation=StorageRoute.sql(alias), execution=step, checkpoint=step),
            coordination=StateTurnResources(),
        )
        async with Persistence.open("split-alias", storage=storage) as state:
            execution_group = state._stores[PersistenceDomain.EXECUTION].storage_group
            conversation_group = state._stores[PersistenceDomain.CONVERSATION].storage_group
            assert execution_group is not conversation_group
            assert execution_group.context.engine is source
            assert conversation_group.context.engine is alias
            assert state._configured_split_step_source() is None
            left = await execution_group._existing_archive_scope_id()
            right = await conversation_group._existing_archive_scope_id()
            assert left == right
            assert (left is not None) == initialized


async def _assert_split_step_text_roundtrip(storage: PersistenceStorage) -> None:
    namespace = "split-text"
    principal = Principal("reader", "default")
    policy = _SessionPolicy(principal.tenant_id)
    model_calls = 0

    async def reply(_messages, _info):
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("retained in the source database")])

    features = RuntimeFeatures(namespace)
    features.capability(SessionCapability(checkpoints=True))
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open(namespace, storage=storage) as state:
        async with Runtime.open(
            persistence=state, features=model_features(_Models(_UsageFunctionModel(reply)), features),
            authorization=policy,
        ) as runtime:
            await runtime.sessions.create_turn_session("chat", principal=principal)
            frames = asyncio.Queue()
            detach = runtime.sessions.observe_turn_session("chat", frames.put_nowait)
            try:
                turn_id = await runtime.sessions.start_turn(
                    "chat", "question", agent="default", principal=principal, idempotency_key="first",
                )
                async with asyncio.timeout(10):
                    while (frame := await frames.get())["type"] != "terminal":
                        pass
                assert frame["turn_id"] == turn_id and frame["state"] == "succeeded", frame
            finally:
                detach()

            committed = await state.session_turns.get_turn(SessionTurnKey("chat", turn_id))
            assert committed is not None and committed["split_source_receipt"] is not None
            assert await state.execution.executions.get(turn_id, tenant_id=principal.tenant_id) is None

    async with Persistence.open(namespace, storage=storage, read_only=True) as state:
        async with History.open(persistence=state, authorization=policy) as history:
            detail = await history.turn_result("chat", turn_id, principal=principal)
            assert detail["turn_id"] == turn_id and detail["conversation_committed"]
            assert detail["output"] == {"text": "retained in the source database"}
            page = await history.turn_session("chat", principal=principal, limit=1)
            assert page["timeline"]["items"] == [detail]
        assert await state.execution.executions.get(turn_id, tenant_id=principal.tenant_id) is None
    assert model_calls == 1
