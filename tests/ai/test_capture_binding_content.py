#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen input captures share exact admitted bindings, not mutable declarations."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive, PreparedArchive
from linktools.ai.persistence._capture_binding_content import (
    capture_binding_references, expand_capture_bindings, map_capture_binding_slots,
    prepare_capture_bindings, restore_capture_bindings,
)
from linktools.ai.persistence.codec import (
    _encode_persisted_domain, decode_domain, encode_agent_binding_content, encode_domain,
)
from linktools.ai.task import TaskRef
from linktools.ai.task._invocation_input import TaskInvocationInputContract

from .test_runtime_persistence_evidence import _execution


@pytest.mark.asyncio
async def test_capture_contract_shares_execution_bytes_and_distinguishes_same_revision(tmp_path: Path) -> None:
    binding = _execution().binding
    binding = replace(binding, agent_spec=replace(binding.agent_spec, system_prompt="capture " * 7000))
    other = replace(binding, agent_spec=replace(binding.agent_spec, system_prompt="different"))
    assert binding.agent_spec.revision == other.agent_spec.revision
    key = "v1/input-capture/agent/test"
    async with Persistence.open("capture-bindings", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "capture.db")) as state:
        store = state.task.tasks.state_store
        scope, archive_store = await store.storage_group.open_archive(create=True)
        archive = ContentArchive(archive_store, scope, namespace="capture-bindings", tenant_id="tenant", object_store=None)
        values = [{"payload": {"binding": item.to_payload(), "unknown": {"keep": True}}} for item in (binding, binding, other)]
        prepared = PreparedArchive()
        additive = {"payload": {"binding": {**binding.to_payload(), "subagent_bindings": []}}}
        assert await prepare_capture_bindings(key, additive, archive, prepared) == additive
        projected = [await prepare_capture_bindings(key, value, archive, prepared) for value in values]
        await archive_store.mutate(lambda tx: archive.publish(tx, prepared))
        refs = [capture_binding_references(key, value)[0] for value in projected]
        assert refs[0] == refs[1] and refs[0] != refs[2]
        raw, = await archive.read_contents((refs[0][2],), codec="execution-binding")
        assert raw == canonical_json_bytes(_encode_persisted_domain(binding)) == encode_agent_binding_content(binding)
        assert len(canonical_json_bytes(projected[0])) < len(canonical_json_bytes(values[0])) // 10
        for actual, expected in zip(projected, values, strict=True):
            assert await restore_capture_bindings(key, actual, archive) == expected
        assert set(projected[0]["payload"]["binding"]) == {
            "$execution_binding", "archive_scope", "content_scope"}
        corrupt = deepcopy(projected[0])
        corrupt["payload"]["binding"]["summary"] = {"id": "another"}
        with pytest.raises(AIError):
            expand_capture_bindings(key, corrupt, {refs[0]: raw})
        # The fixed address selects content; reads do not recompute its digest.
        assert expand_capture_bindings(
            key, projected[0], {refs[0]: encode_agent_binding_content(other)},
        ) == values[2]
        with pytest.raises(AIError):
            expand_capture_bindings(key, projected[0], {})


def test_capture_binding_mapper_keeps_business_mappings_and_additive_fields() -> None:
    binding = _execution().binding.to_payload()
    contract = TaskInvocationInputContract("source", TaskRef("task", 1), {"binding": binding}, {},
        {"type": "agent", "config": {"binding_contract": binding, "extra": "keep"}, "extension": True})
    value = {"payload": {"binding": binding, "task_input": encode_domain(contract), "prompt": {"binding": binding}}}
    changed = map_capture_binding_slots("v1/input-capture/agent/test", value, lambda _binding: {"marker": True})
    assert changed["payload"]["binding"] == {"marker": True}
    assert changed["payload"]["prompt"] == value["payload"]["prompt"]
    restored = decode_domain(changed["payload"]["task_input"], TaskInvocationInputContract)
    assert dict(restored.input) == dict(contract.input)
    assert restored.binding["extension"] is True
    assert restored.binding["config"]["extra"] == "keep"
    assert restored.binding["config"]["binding_contract"] == {"marker": True}
    assert map_capture_binding_slots("v1/input-capture/expired/test", value, lambda _: None) == value
