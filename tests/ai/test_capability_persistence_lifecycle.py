#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence follows native capability replacements without retargeting accepted writes."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from copy import copy
from hashlib import sha256
from typing import Any
from types import SimpleNamespace

import pytest
from pydantic_ai import Agent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.tools import RunContext

from linktools.ai.capability import AgentContext
from linktools.ai.core import CommitPolicy, JsonValue
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence.capabilities import CapabilityWriter, RuntimeCapabilityWrites
from linktools.ai.persistence.state import StateTransaction, StoredRecord
from linktools.ai.runtime._capabilities import _PersistenceCapability

from ._runtime_test_helpers import _UsageFunctionModel


class _ReplacementPersistence(AbstractCapability[AgentContext[None]]):
    id = "probe"

    def __init__(self, *, replace_agent: bool, replace_run: bool) -> None:
        self.commit = CommitPolicy.terminal()
        self.replace_agent = replace_agent
        self.replace_run = replace_run
        self.writer: CapabilityWriter | None = None
        self.owner = "template"
        self.bound: list[_ReplacementPersistence] = []
        self.hooks: list[tuple[str, str]] = []

    def bind_persistence(self, writer: CapabilityWriter) -> _ReplacementPersistence:
        bound = copy(self)
        bound.writer = writer
        bound.owner = "bound"
        self.bound.append(bound)
        return bound

    def for_agent(self, agent: AbstractAgent[AgentContext[None], Any]) -> _ReplacementPersistence:
        if not self.replace_agent:
            return self
        replacement = copy(self)
        replacement.owner = "agent"
        return replacement

    async def for_run(self, ctx: RunContext[AgentContext[None]]) -> _ReplacementPersistence:
        self.hooks.append((ctx.deps.execution_id, "for_run"))
        await self.submit(ctx, "early")
        if not self.replace_run:
            return self
        replacement = copy(self)
        replacement.owner = ctx.deps.execution_id
        return replacement

    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        assert ctx.capabilities[self.id] is self
        self.hooks.append((ctx.deps.execution_id, "before_run"))
        await self.submit(ctx, "late")

    async def submit(self, ctx: RunContext[AgentContext[None]], phase: str) -> None:
        assert self.writer is not None
        key = f"{ctx.deps.execution_id}:{phase}"
        await self.writer.submit(key=key, payload={"key": key, "expected_owner": self.owner})

    async def persist_in_transaction(self, transaction: StateTransaction, payload: Mapping[str, JsonValue]) -> None:
        key = sha256(str(payload["key"]).encode()).digest()
        await transaction.insert_record(StoredRecord(
            key, None, None, "probe", str(payload["key"]), None, 0, None, 0, None,
            {**payload, "actual_owner": self.owner},
        ))


@pytest.mark.asyncio
@pytest.mark.parametrize(("replace_agent", "replace_run"), [(True, False), (False, True), (True, True)])
async def test_native_persistence_replacements_keep_current_and_accepted_write_owners(
    replace_agent: bool, replace_run: bool,
) -> None:
    template = _ReplacementPersistence(replace_agent=replace_agent, replace_run=replace_run)
    model_entered: asyncio.Queue[None] = asyncio.Queue()
    release = asyncio.Event()

    async def model(messages, info):
        model_entered.put_nowait(None)
        await release.wait()
        return ModelResponse(parts=[TextPart("done")])

    store = InMemoryStateStore()
    await store.initialize()
    writes = RuntimeCapabilityWrites(
        {"probe": store}, namespace="persistence-lifecycle", tenant_id="tenant",
        durable_capability_ids=frozenset(), require_active_run=lambda run_id, execution_id: None,
        has_active_execution=lambda execution_id: True,
    )
    agent = Agent(_UsageFunctionModel(model))
    executions = ("one", "two")
    bound = [writes.bind("probe", name, name, template, revision=7, adapt=_PersistenceCapability) for name in executions]
    try:
        runs = [asyncio.create_task(agent.run(
            "question", deps=SimpleNamespace(execution_id=name), capabilities=[capability],
        )) for name, capability in zip(executions, bound)]
        try:
            await asyncio.wait_for(model_entered.get(), 5)
            await asyncio.wait_for(model_entered.get(), 5)
        finally:
            release.set()
        outcomes = await asyncio.gather(*runs)
        assert all(outcome.output == "done" for outcome in outcomes)
        await writes.close()
        rows = await store.read(lambda transaction: transaction.scan_records())
        assert len(rows) == 4
        assert [(row.data["key"], row.data["actual_owner"]) for row in rows] == [
            (row.data["key"], row.data["expected_owner"]) for row in rows
        ]
        assert sorted(template.hooks) == sorted(
            (execution, hook) for execution in executions for hook in ("for_run", "before_run")
        )
        assert len(template.bound) == 2
        assert template.bound[0].writer is not template.bound[1].writer
        assert template.writer is None and template.owner == "template"
    finally:
        await writes.close()
        await store.close()
