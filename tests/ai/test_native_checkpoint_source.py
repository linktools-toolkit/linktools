#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Checkpoint backup sources describe the actual admitted native owner."""

from pathlib import Path

import pytest
from pydantic_ai_harness.step_persistence import InMemoryStepStore

from linktools.ai.steps import native_store_key
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_domain, encode_domain
from linktools.ai.steps import NativeStepSource, rekey_native_step_sources


def test_native_source_codec_relocation_preserves_partition_and_application_data() -> None:
    source = NativeStepSource("a" * 32, native_store_key("b", "tenant"))
    encoded = encode_domain(source)
    assert decode_domain(encoded, NativeStepSource) == source
    original = {"source": encoded, "application": {"group_id": source.group_id}}
    relocated = rekey_native_step_sources(original, {source.group_id: "c" * 32})
    assert decode_domain(relocated["source"], NativeStepSource) == NativeStepSource("c" * 32, source.store_key)
    assert relocated["application"] == original["application"]
    assert decode_domain(original["source"], NativeStepSource) == source
    with pytest.raises(ValueError):
        rekey_native_step_sources(encoded, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("external_effects", (False, True))
async def test_native_source_requires_builtin_colocated_effect_owner(tmp_path: Path, external_effects: bool) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("source", storage=storage, tenant_id="tenant") as persistence:
        native = await persistence.create_step_store(
            tool_effect_store=InMemoryStepStore() if external_effects else None,
        )
        runs = await persistence.create_agent_run_store(step_store=native)
        try:
            source = runs.native_step_source
            if external_effects:
                assert source is None
                assert native.backup_source is None
            else:
                assert source is not None
                assert source == native.backup_source
                runs.validate_native_step_source(source)
                with pytest.raises(AIError) as rejected:
                    runs.validate_native_step_source(NativeStepSource("0" * 32, source.store_key))
                assert rejected.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
        finally:
            await runs.preflight_close()
            await runs.close()
            await native.close()
    if not external_effects:
        async with Persistence.open("source", storage=storage, tenant_id="tenant", read_only=True) as offline:
            owner = await offline.checkpoint_backup_source(source)
            try:
                assert owner.backup_source == source
                with pytest.raises(AIError) as rejected:
                    await offline.checkpoint_backup_source(NativeStepSource(source.group_id, native_store_key("0", "tenant")))
                assert rejected.value.code is ErrorCode.STORAGE_OWNER_MISMATCH
            finally:
                await owner.close()


@pytest.mark.asyncio
async def test_custom_native_owner_is_not_claimed_as_builtin_backup(tmp_path: Path) -> None:
    async with Persistence.open("source", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        runs = await persistence.create_agent_run_store(step_store=InMemoryStepStore())
        try:
            assert runs.native_step_source is None
        finally:
            await runs.preflight_close()
            await runs.close()
