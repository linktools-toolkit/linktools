#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One Runtime opening contract with explicit storage lifetime ownership."""

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime


@pytest.mark.asyncio
async def test_open_owned_storage_retains_existing_namespace_and_tenant_contract() -> None:
    runtime = Runtime.open("owned-runtime", storage=PersistenceStorage.in_memory(), tenant_id="tenant")
    async with runtime:
        state = runtime._state
        assert runtime.namespace == "owned-runtime" and runtime.tenant_id == "tenant"
        assert state.ready
    assert not state.ready
    await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["open", "attach"])
async def test_open_and_compatibility_attach_borrow_without_closing(entry: str) -> None:
    async with Persistence.open("borrowed-runtime", storage=PersistenceStorage.in_memory(), tenant_id="owner") as state:
        async with getattr(Runtime, entry)(persistence=state) as runtime:
            assert runtime._state is state
            assert runtime.namespace == "borrowed-runtime" and runtime.tenant_id == "owner"
        assert state.ready
        # Successful cleanup released precisely this Runtime's owner binding.
        await state.close()
        assert not state.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"namespace": "borrowed-runtime"},
    {"namespace": "other"},
    {"tenant_id": "default"},
    {"tenant_id": "other"},
    {"storage": PersistenceStorage.in_memory()},
])
async def test_borrowed_open_rejects_storage_and_identity_overrides(overrides) -> None:
    async with Persistence.open("borrowed-runtime", storage=PersistenceStorage.in_memory()) as state:
        with pytest.raises(ValueError):
            Runtime.open(persistence=state, **overrides)
        assert state.ready


@pytest.mark.asyncio
async def test_borrowed_open_rechecks_state_at_enter() -> None:
    async with Persistence.open("enter-check", storage=PersistenceStorage.in_memory()) as state:
        runtime = Runtime.open(persistence=state)
        await state.close()
        with pytest.raises(AIError) as error:
            await runtime.__aenter__()
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        await runtime.close()


@pytest.mark.asyncio
async def test_borrowed_open_rejects_unopened_and_closed_state() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    with pytest.raises(AIError) as error:
        Runtime.open(persistence=state)
    assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
    await state.initialize(namespace="closed-check", tenant_id="default")
    await state.close()
    with pytest.raises(AIError) as error:
        Runtime.open(persistence=state)
    assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY


@pytest.mark.asyncio
async def test_borrowed_open_rejects_read_only_persistence(tmp_path) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "runtime.db")
    async with Persistence.open("read-only", storage=storage):
        pass
    async with Persistence.open("read-only", storage=storage, read_only=True) as state:
        with pytest.raises(AIError) as error:
            Runtime.open(persistence=state)
        assert error.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
        assert state.ready


def test_open_requires_explicit_storage_or_persistence() -> None:
    with pytest.raises(ValueError):
        Runtime.open()
    with pytest.raises(ValueError):
        Runtime.open("namespace")
    with pytest.raises(ValueError):
        Runtime.open(storage=PersistenceStorage.in_memory())
