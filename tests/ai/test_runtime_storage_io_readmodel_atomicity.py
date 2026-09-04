#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ReadModel transaction atomicity contracts for storage I/O batching."""

from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

import pytest
from linktools.ai.runtime import RuntimeState
from linktools.ai.runtime.state import FactQuery
from linktools.ai.runtime.state._readmodel import (
    ExecutionReadModelBuild,
    ExecutionReadModelRepository,
    ExecutionReadModelStatus,
)
from linktools.ai.runtime.state._store import (
    StateStore,
    StateTransaction,
    StoredFact,
)

pytestmark = pytest.mark.asyncio

ResultT = TypeVar("ResultT")


class _FailingFactsStore:
    def __init__(self, delegate: StateStore, *, fail_on_insert: int) -> None:
        self._delegate = delegate
        self._fail_on_insert = fail_on_insert
        self.insert_facts_count = 0

    async def read(
        self,
        operation: Callable[[StateTransaction], Awaitable[ResultT]],
    ) -> ResultT:
        return await self._delegate.read(operation)

    async def mutate(
        self,
        operation: Callable[[StateTransaction], Awaitable[ResultT]],
    ) -> ResultT:
        async def wrapped(transaction: StateTransaction) -> ResultT:
            proxy = _FailingFactsTransaction(transaction, self)
            return await operation(proxy)  # type: ignore[arg-type]

        return await self._delegate.mutate(wrapped)


class _FailingFactsTransaction:
    def __init__(self, delegate: StateTransaction, owner: _FailingFactsStore) -> None:
        self._delegate = delegate
        self._owner = owner

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    async def insert_facts(self, facts: Sequence[StoredFact]) -> None:
        self._owner.insert_facts_count += 1
        if self._owner.insert_facts_count == self._owner._fail_on_insert:
            raise RuntimeError("injected final fact failure")
        await self._delegate.insert_facts(facts)


async def test_final_read_model_batch_failure_does_not_publish_complete() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="io-readmodel-rollback", tenant_id="tenant")
    try:
        delegate = state.execution.executions.state_store
        store = _FailingFactsStore(delegate, fail_on_insert=2)
        repository = ExecutionReadModelRepository(
            store,  # type: ignore[arg-type]
            namespace="io-readmodel-rollback",
            tenant_id="tenant",
        )
        owner, fence, claimed = await repository._claim("execution")
        assert claimed is None
        build = ExecutionReadModelBuild(
            "execution",
            "tenant",
            "source",
            tuple({"index": index} for index in range(129)),
            (),
            (),
        )

        with pytest.raises(RuntimeError, match="injected final fact failure"):
            await repository._write_build(build, owner, fence)

        assert store.insert_facts_count == 2
        stored = await delegate.read(
            lambda transaction: transaction.get_record(
                repository._record_key("execution")
            )
        )
        assert stored is not None
        value = repository._decode_record(stored)
        assert value.status is ExecutionReadModelStatus.BUILDING
        assert stored.lease_owner == owner
        assert stored.lease_fence == fence
        assert await repository.get_complete("execution", tenant_id="tenant") is None

        facts = await delegate.read(
            lambda transaction: transaction.list_facts(
                FactQuery(repository._stream("execution", "trace"))
            )
        )
        assert len(facts) == 1
        assert len(facts[0].data["items"]) == 128
    finally:
        await state.close()
