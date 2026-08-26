#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for opaque capability semantic identity."""

from dataclasses import dataclass

import pytest
from linktools.ai.capability import CapabilityGroup, RunContext
from linktools.ai.errors import AIError, ErrorCode
from pydantic_ai.capabilities import AbstractCapability


@dataclass
class _Capability(AbstractCapability[RunContext[None]]):
    id: str = "test-capability"

    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return None


@pytest.mark.asyncio
async def test_capability_revision_is_fingerprint_input_only() -> None:
    first = CapabilityGroup[None]("first")
    first.capability(_Capability(), revision=1)
    second = CapabilityGroup[None]("second")
    second.capability(_Capability(), revision=2)

    first_candidate = (await first.freeze())[0]
    second_candidate = (await second.freeze())[0]

    assert first_candidate.kind == "capability"
    assert first_candidate.id == "test-capability"
    assert first_candidate.semantic_revision == 1
    assert second_candidate.semantic_revision == 2
    assert first_candidate.fingerprint != second_candidate.fingerprint
    assert first_candidate.semantic_contract["semantic_revision"] == 1
    assert second_candidate.semantic_contract["semantic_revision"] == 2
    assert "restore_locator" not in first_candidate.semantic_contract


@pytest.mark.asyncio
async def test_identical_capability_semantics_have_stable_fingerprint() -> None:
    left = CapabilityGroup[None]("left")
    left.capability(_Capability(), revision=7)
    right = CapabilityGroup[None]("right")
    right.capability(_Capability(), revision=7)

    left_candidate = (await left.freeze())[0]
    right_candidate = (await right.freeze())[0]

    assert left_candidate.fingerprint == right_candidate.fingerprint
    assert left_candidate.semantic_contract == right_candidate.semantic_contract


@pytest.mark.parametrize("revision", [0, -1, True])
def test_invalid_capability_revision_is_rejected(revision: object) -> None:
    group = CapabilityGroup[None]("group")
    with pytest.raises(AIError) as error:
        group.capability(_Capability(), revision=revision)  # type: ignore[arg-type]
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_duplicate_capability_identity_is_rejected_when_group_freezes() -> None:
    group = CapabilityGroup[None]("group")
    group.capability(_Capability(), revision=1)
    group.capability(_Capability(), revision=1)

    with pytest.raises(AIError) as error:
        await group.freeze()

    assert error.value.code is ErrorCode.CAPABILITY_CONFLICT
