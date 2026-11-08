#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Idempotency repository target and conditional update contracts."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from linktools.ai.core import IdempotencyStatus, ResourceKind
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import IdempotencyRecord
from linktools.ai.persistence import Persistence, PersistenceStorage


@pytest.fixture(params=("memory", "sqlite"))
def storage(request: pytest.FixtureRequest, tmp_path: Path) -> PersistenceStorage:
    if request.param == "memory":
        return PersistenceStorage.in_memory()
    return PersistenceStorage.sqlite(tmp_path / "idempotency.sqlite")


def _record(resource_kind: ResourceKind) -> IdempotencyRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return IdempotencyRecord(
        scope="run", idempotency_key_digest="a" * 64, request_digest="1" * 64,
        resource_kind=resource_kind, resource_id="target",
        status=IdempotencyStatus.RESERVED, result_digest=None, error_code=None,
        created_at=now, updated_at=now,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_kind", (ResourceKind.EXECUTION, ResourceKind.EVALUATION))
@pytest.mark.parametrize("identity_component", ("scope", "key"))
async def test_idempotency_compare_and_swap_cannot_redirect_target(
    storage: PersistenceStorage, resource_kind: ResourceKind, identity_component: str,
) -> None:
    async with Persistence.open("idempotency", storage=storage, tenant_id="tenant") as persistence:
        repository = (persistence.execution.idempotency if resource_kind is ResourceKind.EXECUTION
                      else persistence.evaluation.idempotency)
        target = _record(resource_kind)
        other = replace(
            target,
            scope="other" if identity_component == "scope" else target.scope,
            idempotency_key_digest="b" * 64 if identity_component == "key" else target.idempotency_key_digest,
            resource_id="other", status=IdempotencyStatus.FAILED,
        )
        await repository.reserve(target)
        await repository.reserve(other)

        with pytest.raises(AIError) as error:
            await repository.compare_and_swap(
                target.scope, target.idempotency_key_digest, tenant_id="tenant",
                expected_status=IdempotencyStatus.RESERVED,
                next_record=replace(other, status=IdempotencyStatus.STARTED),
            )

        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        assert await repository.get(target.scope, target.idempotency_key_digest, tenant_id="tenant") == target
        assert await repository.get(other.scope, other.idempotency_key_digest, tenant_id="tenant") == other


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_kind", (ResourceKind.EXECUTION, ResourceKind.EVALUATION))
async def test_idempotency_compare_and_swap_updates_matching_target(
    storage: PersistenceStorage, resource_kind: ResourceKind,
) -> None:
    async with Persistence.open("idempotency", storage=storage, tenant_id="tenant") as persistence:
        repository = (persistence.execution.idempotency if resource_kind is ResourceKind.EXECUTION
                      else persistence.evaluation.idempotency)
        target = _record(resource_kind)
        await repository.reserve(target)
        candidate = replace(
            target, status=IdempotencyStatus.COMPLETED, result_digest="c" * 64,
            updated_at=target.updated_at + timedelta(seconds=1),
        )

        actual = await repository.compare_and_swap(
            target.scope, target.idempotency_key_digest, tenant_id="tenant",
            expected_status=IdempotencyStatus.RESERVED, next_record=candidate,
        )

        assert actual == candidate
        assert await repository.get(target.scope, target.idempotency_key_digest, tenant_id="tenant") == candidate


@pytest.mark.asyncio
@pytest.mark.parametrize("resource_kind", (ResourceKind.EXECUTION, ResourceKind.EVALUATION))
@pytest.mark.parametrize("failure", ("status", "tenant", "kind"))
async def test_idempotency_compare_and_swap_preserves_existing_preconditions(
    storage: PersistenceStorage, resource_kind: ResourceKind, failure: str,
) -> None:
    async with Persistence.open("idempotency", storage=storage, tenant_id="tenant") as persistence:
        repository = (persistence.execution.idempotency if resource_kind is ResourceKind.EXECUTION
                      else persistence.evaluation.idempotency)
        target = _record(resource_kind)
        await repository.reserve(target)
        candidate = replace(target, status=IdempotencyStatus.STARTED)
        expected_status = IdempotencyStatus.RESERVED
        tenant_id = "tenant"
        expected_error = ErrorCode.STORAGE_CONFLICT
        if failure == "status":
            expected_status = IdempotencyStatus.FAILED
        elif failure == "tenant":
            tenant_id = "other"
            expected_error = ErrorCode.STORAGE_OWNER_MISMATCH
        else:
            candidate = replace(candidate, resource_kind=ResourceKind.ARTIFACT)
            expected_error = ErrorCode.STORAGE_INTEGRITY_ERROR

        with pytest.raises(AIError) as error:
            await repository.compare_and_swap(
                target.scope, target.idempotency_key_digest, tenant_id=tenant_id,
                expected_status=expected_status, next_record=candidate,
            )

        assert error.value.code is expected_error
        assert await repository.get(target.scope, target.idempotency_key_digest, tenant_id="tenant") == target
