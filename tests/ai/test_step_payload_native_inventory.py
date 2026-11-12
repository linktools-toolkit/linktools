#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A shared bytes relation does not merge typed payload and Native ownership."""
import hashlib
from pathlib import Path
import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import insert, select
from linktools.ai.errors import AIError
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage
from .test_step_content_cleanup import _Offline
from linktools.ai.persistence._step_maintenance import native_message_partitions
from linktools.ai.persistence._step_payload_owner import payload_store_key, payload_store_locator
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.steps import native_store_key, native_store_locator

@pytest.mark.asyncio
async def test_same_digest_payload_partition_is_not_in_native_offline_closure(tmp_path: Path) -> None:
    async with Persistence.open("shared-bytes", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "content.db")) as state:
        native = await state.create_step_store()
        try:
            context = state.execution.executions.state_store.storage_group.context
            table = build_step_sql_metadata().tables["ai_step_messages"]
            raw = encode_model_messages((ModelResponse(parts=[TextPart("same bytes")]),))
            digest = hashlib.sha256(raw).hexdigest()
            model_key = native_store_key("shared-bytes", "tenant")
            payload_key = payload_store_key("shared-bytes", "tenant")
            async with context.sessions() as session:
                await session.execute(insert(table), [
                    {"store_key": key, "store_locator": locator, "sha256": digest, "message": raw,
                     "compression": "raw", "raw_size": len(raw), "payload_json": None}
                    for key, locator in ((model_key, native_store_locator(model_key)), (payload_key, payload_store_locator(payload_key)))])
                assert await native_message_partitions(context, session, table) == (model_key,)
                await session.commit()
            await native.purge_unreferenced_content(exclusive=_Offline())
            async with context.sessions() as session:
                assert (await session.execute(select(table.c.store_key, table.c.sha256))).all() == [(payload_key, digest)]
        finally:
            await native.close()

@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("tuple", "locator"))
async def test_native_inventory_rejects_unknown_or_malformed_partition(tmp_path: Path, damage: str) -> None:
    async with Persistence.open("shared-bytes", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "invalid.db")) as state:
        native = await state.create_step_store()
        try:
            context = state.execution.executions.state_store.storage_group.context
            table = build_step_sql_metadata().tables["ai_step_messages"]
            key = b"unknown" if damage == "tuple" else payload_store_key("shared-bytes", "tenant")
            async with context.sessions() as session:
                await session.execute(insert(table).values(store_key=key, store_locator="0" * 64,
                    sha256=hashlib.sha256(b"body").hexdigest(), message=b"body", compression="raw", raw_size=4, payload_json=None))
                with pytest.raises(AIError):
                    await native_message_partitions(context, session, table)
                await session.rollback()
        finally:
            await native.close()
