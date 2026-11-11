#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run metadata has one owner independently of optional Step capture."""

import asyncio
from dataclasses import replace
from datetime import timedelta

import pytest
from pydantic_ai_harness.step_persistence import InMemoryStepStore, RunRecord

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._run_registration import RunRegistrationStore, RunRegistrationReference
from linktools.ai.persistence._official_step_reader import OfficialStepReader


@pytest.mark.asyncio
async def test_confirmed_external_registration_survives_external_store_lifetime() -> None:
    state = InMemoryStateStore()
    await state.initialize()
    external = InMemoryStepStore()
    source = RunRecord(run_id="run", registration_id="registration", metadata={"source": "native"})
    await external.register_run(source)
    confirmed = await external.get_run(run_id="run")
    assert confirmed is not None
    owner = RunRegistrationStore(state, namespace="owner", tenant_id="tenant")
    await owner.publish(confirmed)
    external = InMemoryStepStore()
    assert await external.get_run(run_id="run") is None
    reopened = RunRegistrationStore(state, namespace="owner", tenant_id="tenant")
    assert await reopened.get("run") == source


@pytest.mark.asyncio
async def test_domains_import_one_equal_registration_and_reject_conflicts() -> None:
    state = InMemoryStateStore()
    await state.initialize()
    owners = [RunRegistrationStore(state, namespace="owner", tenant_id="tenant") for _ in range(3)]
    source = RunRecord(run_id="run", registration_id="registration", metadata={"shared": "value"})
    await asyncio.gather(*(owner.publish(source) for owner in owners))
    for owner in owners:
        assert await owner.get("run") == source
    with pytest.raises(AIError) as caught:
        await owners[0].publish(replace(source, metadata={"shared": "conflict"}))
    assert caught.value.code is ErrorCode.STORAGE_CONFLICT
    assert await owners[0].get("run") == source


@pytest.mark.asyncio
async def test_run_registration_is_partitioned_by_namespace_and_tenant() -> None:
    state = InMemoryStateStore()
    await state.initialize()
    owner = RunRegistrationStore(state, namespace="owner", tenant_id="tenant")
    await owner.publish(RunRecord(run_id="run"))
    assert await RunRegistrationStore(state, namespace="other", tenant_id="tenant").get("run") is None
    assert await RunRegistrationStore(state, namespace="owner", tenant_id="other").get("run") is None


def test_domain_reference_stores_only_projection_differences() -> None:
    source = RunRecord(run_id="run", registration_id="registration", metadata={"same": "large" * 1000,
                        "different": "source", "removed": "value"})
    projected = replace(OfficialStepReader.project_run(source),
                        metadata={"same": source.metadata["same"], "different": "domain", "added": "value"},
                        started_at=source.started_at + timedelta(microseconds=1))
    reference = RunRegistrationReference.capture(source, projected)
    assert reference.metadata == {"different": "domain", "added": "value"}
    assert reference.removed_metadata == ("removed",)
    assert reference.project(source) == projected
    with pytest.raises(AIError) as caught:
        reference.project(replace(source, registration_id="another"))
    assert caught.value.code is ErrorCode.STORAGE_OWNER_MISMATCH


def test_equal_projection_has_no_duplicate_fields_or_metadata() -> None:
    source = RunRecord(run_id="run", metadata={"shared": "value"})
    reference = RunRegistrationReference.capture(source, OfficialStepReader.project_run(source))
    assert reference.fields == {} and reference.metadata == {} and reference.removed_metadata == ()
    assert reference.project(source) == OfficialStepReader.project_run(source)
