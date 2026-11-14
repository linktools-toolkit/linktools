#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Custom native capability trees retain their declared persistence participants."""

from collections.abc import Callable, Mapping
from copy import copy

import pytest
from pydantic_ai.capabilities import AbstractCapability, CombinedCapability, WrapperCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.tools import RunContext

from linktools.ai.capability import AgentContext, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.capabilities import CapabilityWriter
from linktools.ai.persistence.state import StateTransaction, StoredRecord
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _RunNote(AbstractCapability[AgentContext[None]]):
    id = "custom.note"
    commit = CommitPolicy.sync()
    writer: CapabilityWriter | None = None

    def bind_persistence(self, writer: CapabilityWriter) -> "_RunNote":
        bound = copy(self)
        bound.writer = writer
        return bound

    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        assert self.writer is not None
        await self.writer.submit(key="started", payload={"execution_id": ctx.deps.execution_id})

    async def persist_in_transaction(self, transaction: StateTransaction, payload: Mapping[str, JsonValue]) -> None:
        await transaction.insert_record(StoredRecord(
            key=b"started", scope_key=None, parent_key=None, kind="custom-note",
            sort_key=b"started", state=None, storage_revision=0,
            lease_owner=None, lease_fence=0, lease_expires_at=None, data=payload,
        ))


class _NoteGroup(AbstractCapability[AgentContext[None]]):
    def __init__(self, note: AbstractCapability[AgentContext[None]]) -> None:
        self.note = note

    def apply(self, visitor: Callable[[AbstractCapability[AgentContext[None]]], None]) -> None:
        self.note.apply(visitor)

    def visit_and_replace(
        self, visitor: Callable[[AbstractCapability[AgentContext[None]]], AbstractCapability[AgentContext[None]] | None],
    ) -> "_NoteGroup | None":
        note = self.note.visit_and_replace(visitor)
        if note is None:
            return None
        if note is self.note:
            return self
        bound = copy(self)
        bound.note = note
        return bound

    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        await self.note.before_run(ctx)


@pytest.mark.asyncio
@pytest.mark.parametrize("container", ("custom", "wrapped_custom", "wrapped_combined"))
async def test_native_container_persists_its_registered_leaf(container: str) -> None:
    async def model(messages, info):
        return ModelResponse(parts=[TextPart("done")])

    note = _RunNote()
    if container == "wrapped_combined":
        group = WrapperCapability(CombinedCapability([note]))
    elif container == "wrapped_custom":
        group = WrapperCapability(_NoteGroup(note))
    else:
        group = _NoteGroup(note)
    features = RuntimeFeatures("custom-container")
    features.agent("default", model="default", allow_tools=())
    features.capability(group, id="custom.notes", revision=3, scope="runtime")
    async with Persistence.open("custom-container", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(
            persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features),
        ) as runtime:
            execution = await runtime.agents.get().start("question")
            result = (await execution.wait(timeout_seconds=5)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            store = await persistence.capability_store(note.id)
            rows = await store.read(lambda transaction: transaction.scan_records())
            assert len(rows) == 1
            assert rows[0].data == {"execution_id": execution.execution_id}
            assert note.writer is None


@pytest.mark.asyncio
async def test_custom_native_container_rejects_duplicate_persistent_ids() -> None:
    features = RuntimeFeatures("duplicate-container")
    features.capability(
        _NoteGroup(CombinedCapability([_RunNote(), _RunNote()])),
        id="custom.notes", revision=3, scope="runtime",
    )
    async with Persistence.open("duplicate-container", storage=PersistenceStorage.in_memory()) as persistence:
        with pytest.raises(AIError) as conflict:
            async with Runtime.attach(persistence=persistence, features=features):
                pytest.fail("duplicate persistent capability IDs were accepted")
        assert conflict.value.code is ErrorCode.CAPABILITY_CONFLICT


@pytest.mark.asyncio
async def test_custom_native_container_rejects_duplicate_persistence_domains() -> None:
    first, second = TraceCapability(), TraceCapability()
    first.id, second.id = "trace.first", "trace.second"
    features = RuntimeFeatures("duplicate-domains")
    features.capability(_NoteGroup(CombinedCapability([first, second])), id="custom.traces", scope="runtime")
    async with Persistence.open("duplicate-domains", storage=PersistenceStorage.in_memory()) as persistence:
        with pytest.raises(AIError) as conflict:
            async with Runtime.attach(persistence=persistence, features=features):
                pytest.fail("duplicate persistent domain declarations were accepted")
        assert conflict.value.code is ErrorCode.CAPABILITY_CONFLICT
