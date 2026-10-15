#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Child metadata reads retain authorization and consistent read boundaries."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

from linktools.ai.core import ExecutionLineageKind
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._execution_repositories import ExecutionRepositoryImpl

from .test_execution_result_read_io import _PRINCIPAL, _seed, _service
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_children_read_observes_authorization_changes_outside_read_scope(
    tmp_path: Path, backend: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="children-read", tenant_id="tenant")
    try:
        parent = await _seed(state)
        child = replace(parent, execution_id="child", parent_execution_id="execution",
                        parent_invocation_id="invocation", lineage_kind=ExecutionLineageKind.SUBAGENT)

        class Authorization:
            async def authorize(self, principal, action, resource):
                # A read-only callback scope must not enclose application writes.
                await state.execution.executions.create_with_history_head(child)

        children = await _service(state, authorization=Authorization()).list_children(
            "execution", principal=_PRINCIPAL,
        )
        assert [value.execution_id for value in children] == ["child"]
        assert children[0].parent_execution_id == "execution"
    finally:
        await state.close()


async def test_children_metadata_uses_fewer_sql_read_setups_without_skipping_queries(
    tmp_path: Path,
) -> None:
    state = _storage("sqlite", tmp_path)
    await state.initialize(namespace="children-read", tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions

        class IndependentOwner(ExecutionRepositoryImpl):
            pass

        independent = IndependentOwner(repository.state_store, namespace="children-read", tenant_id="tenant")
        engine = repository.state_store.storage_group.context.engine.sync_engine
        observations = []
        for owner in (independent, repository):
            statements = []

            def capture(connection, cursor, statement, parameters, context, many):
                statements.append(statement.lstrip().upper())

            event.listen(engine, "before_cursor_execute", capture)
            try:
                assert await _service(state, repository=owner).list_children(
                    "execution", principal=_PRINCIPAL,
                ) == ()
            finally:
                event.remove(engine, "before_cursor_execute", capture)
            observations.append(statements)
        original, grouped = observations
        assert sum(sql.startswith("SELECT") for sql in grouped) == sum(
            sql.startswith("SELECT") for sql in original
        )
        assert sum(sql.startswith("BEGIN") for sql in grouped) < sum(
            sql.startswith("BEGIN") for sql in original
        )
        assert not any(sql.startswith(("INSERT", "UPDATE", "DELETE")) for sql in grouped)
        print("children SQL baseline/grouped:", len(original), len(grouped),
              "BEGIN:", sum(sql.startswith("BEGIN") for sql in original),
              sum(sql.startswith("BEGIN") for sql in grouped))
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_children_read_fails_closed_when_parent_disappears_after_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="children-read", tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions

        class Authorization:
            async def authorize(self, principal, action, resource):
                async def missing(*args, **kwargs):
                    return None
                monkeypatch.setattr(repository, "get", missing)

        with pytest.raises(AIError) as caught:
            await _service(state, authorization=Authorization()).list_children(
                "execution", principal=_PRINCIPAL,
            )
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("denial", ("policy", "tenant"))
async def test_children_read_rejects_unauthorized_parent_before_child_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, denial: str,
) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="children-read", tenant_id="tenant")
    try:
        await _seed(state)
        principal = _PRINCIPAL if denial == "policy" else replace(_PRINCIPAL, tenant_id="other")
        authorizations = []

        class Authorization:
            async def authorize(self, principal, action, resource):
                authorizations.append(resource.id)
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        async def forbidden(*args, **kwargs):
            raise AssertionError("Unauthorized child metadata must not be read")

        monkeypatch.setattr(state.execution.executions, "list_children", forbidden)
        with pytest.raises(AIError) as caught:
            await _service(state, authorization=Authorization()).list_children(
                "execution", principal=principal,
            )
        assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
        assert authorizations == (["execution"] if denial == "policy" else [])
    finally:
        await state.close()


@pytest.mark.parametrize("backend", ("filesystem", "sqlite"))
async def test_children_read_works_with_read_only_storage(tmp_path: Path, backend: str) -> None:
    state = _storage(backend, tmp_path)
    await state.initialize(namespace="children-read", tenant_id="tenant")
    await _seed(state)
    await state.close()
    reader = _storage(backend, tmp_path)
    await reader.initialize(namespace="children-read", tenant_id="tenant", read_only=True)
    try:
        assert await _service(reader).list_children("execution", principal=_PRINCIPAL) == ()
    finally:
        await reader.close()
