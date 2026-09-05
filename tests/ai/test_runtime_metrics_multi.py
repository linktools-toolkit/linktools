#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Metrics integration across RuntimeState backends."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.observe import MetricQuery, MetricWindow, Metrics
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.spec import AgentSpec, AgentSpecCodec
from linktools.ai.workspace import Workspace
from pydantic_ai.models.test import TestModel
from sqlalchemy.ext.asyncio import create_async_engine


class _ModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {"provider": "test", "model": "test"}

    def materialize(self) -> TestModel:
        return TestModel(custom_output_text="ok")


class _Models:
    def snapshot(self) -> "_Models":
        return self

    def resolve(self, route_id: str) -> _ModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _ModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _ModelBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _ModelBinding.semantic_payload
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _ModelBinding()


def _workspace(path: Path) -> Workspace:
    path.mkdir(parents=True)
    agent_path = path / ".linktools" / "agents" / "default"
    agent_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", model="default", allow_tools=())
        )
    )
    return Workspace.load(path)


@pytest.mark.asyncio
async def test_runtime_states_share_metrics_without_lifecycle_coupling(
    tmp_path: Path,
) -> None:
    metrics = Metrics.in_memory(namespace="multi-runtime")
    start = datetime.now(timezone.utc) - timedelta(seconds=1)

    filesystem_state = RuntimeState.filesystem(tmp_path / "filesystem-state")

    sqlite_state_path = tmp_path / "sqlite-state.db"
    sqlite_provision_engine = create_async_engine(
        f"sqlite+aiosqlite:///{sqlite_state_path}"
    )
    await provision_runtime_database(sqlite_provision_engine)
    await sqlite_provision_engine.dispose()
    sqlite_state = RuntimeState.sqlite(sqlite_state_path)

    sql_state_path = tmp_path / "sql-state.db"
    sql_engine = create_async_engine(f"sqlite+aiosqlite:///{sql_state_path}")
    await provision_runtime_database(sql_engine)
    sql_state = RuntimeState.sql(sql_engine)

    cases = (
        (_workspace(tmp_path / "workspace-fs"), filesystem_state, "filesystem"),
        (_workspace(tmp_path / "workspace-sqlite"), sqlite_state, "sqlite"),
        (_workspace(tmp_path / "workspace-sql"), sql_state, "sql"),
    )

    try:
        for workspace, state, label in cases:
            async with Runtime.open(
                workspace,
                models=_Models(),  # type: ignore[arg-type]
                state=state,
                metrics=metrics,
            ) as runtime:
                result = await runtime.agent("default").run(
                    f"hello-{label}",
                    timeout_seconds=10,
                )
                assert result.status is ExecutionStatus.SUCCEEDED

        end = datetime.now(timezone.utc) + timedelta(seconds=1)
        window = MetricWindow.between(start, end)
        model_count = await metrics.query(
            MetricQuery("linktools.model.request.count", window)
        )
        execution_count = await metrics.query(
            MetricQuery("linktools.execution.count", window)
        )

        assert model_count.points[0].value == 3
        assert model_count.points[0].sample_count == 3
        assert execution_count.points[0].value == 3
        assert execution_count.points[0].sample_count == 3
    finally:
        await sql_engine.dispose()
