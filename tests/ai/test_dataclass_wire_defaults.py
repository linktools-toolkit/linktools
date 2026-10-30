#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic wire omissions follow constructor defaults without replacing explicit facts."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import pytest

from linktools.ai.core import BudgetUsage, ResourceKind, ResourceRef, RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence.codec import _CURRENT_CODEC, _decode_domain, _encode_domain, _encode_persisted_domain
from linktools.ai.persistence.contracts import RecoveryCheckpoint, RecoveryCheckpointState, RecoveryHandoffPhase


@dataclass
class _DefaultedRecord:
    required: str
    number: int = 3
    values: list[str] = field(default_factory=list)
    enabled: bool = False
    derived: int = field(default=7, init=False)


def _codec():
    return replace(
        _CURRENT_CODEC,
        wire_ids={**_CURRENT_CODEC.wire_ids, _DefaultedRecord: "defaulted_record"},
        domain_types={**_CURRENT_CODEC.domain_types, "defaulted_record": _DefaultedRecord},
    )


@pytest.mark.parametrize("persisted", (False, True))
def test_defaults_and_mutable_factories_are_applied_by_each_constructor(persisted: bool) -> None:
    codec = _codec()
    payload = _encode_domain(_DefaultedRecord("value", 9, ["explicit"], True), codec, persisted=persisted)
    explicit = _decode_domain(payload, _DefaultedRecord, codec, persisted=persisted)
    assert explicit.number == 9 and explicit.values == ["explicit"] and explicit.enabled is True
    del payload["fields"]["number"]
    del payload["fields"]["values"]
    del payload["fields"]["enabled"]
    first = _decode_domain(payload, _DefaultedRecord, codec, persisted=persisted)
    second = _decode_domain(payload, _DefaultedRecord, codec, persisted=persisted)
    assert first == second == _DefaultedRecord("value")
    first.values.append("local")
    assert second.values == []
    assert "values" not in payload["fields"]


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("missing", ("required", "derived"))
def test_no_default_and_non_init_fields_remain_required(persisted: bool, missing: str) -> None:
    codec = _codec()
    payload = _encode_domain(_DefaultedRecord("value"), codec, persisted=persisted)
    del payload["fields"][missing]
    with pytest.raises(AIError) as caught:
        _decode_domain(payload, _DefaultedRecord, codec, persisted=persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert caught.value.safe_details == {"wire_type": "defaulted_record", "missing_fields": [missing]}


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("field_name,value", (("number", "invalid"), ("derived", 8)))
def test_explicit_invalid_defaulted_or_derived_values_never_fall_back(
    persisted: bool, field_name: str, value: object,
) -> None:
    codec = _codec()
    payload = _encode_domain(_DefaultedRecord("value"), codec, persisted=persisted)
    payload["fields"][field_name] = value
    with pytest.raises(AIError) as caught:
        _decode_domain(payload, _DefaultedRecord, codec, persisted=persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("value,field_name,expected", (
    (RunBudget(total_tokens=100), "total_tokens", RunBudget()),
    (ResourceRef(ResourceKind.SESSION, "session", "tenant", "owner"), "owner_principal_id",
     ResourceRef(ResourceKind.SESSION, "session", "tenant")),
    (RecoveryCheckpoint("execution", "run", RecoveryCheckpointState.COMPLETED, 1,
                        datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc),
                        handoff_phase=RecoveryHandoffPhase.COMPLETED),
     "handoff_phase",
     RecoveryCheckpoint("execution", "run", RecoveryCheckpointState.COMPLETED, 1,
                        datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc))),
))
def test_budget_resource_and_recovery_defaults_preserve_explicit_values(
    persisted: bool, value: object, field_name: str, expected: object,
) -> None:
    payload = _encode_domain(value, _CURRENT_CODEC, persisted=persisted)
    assert _decode_domain(payload, type(value), _CURRENT_CODEC, persisted=persisted) == value
    del payload["fields"][field_name]
    assert _decode_domain(payload, type(value), _CURRENT_CODEC, persisted=persisted) == expected


@pytest.mark.parametrize("value,field_name,invalid", (
    (BudgetUsage("scope", RunBudget(total_tokens=100), total_tokens=21), "total_tokens", -1),
    (ResourceRef(ResourceKind.SESSION, "session", "tenant", "owner"), "owner_principal_id", 7),
))
def test_budget_and_resource_explicit_invalid_values_are_not_replaced(
    value: object, field_name: str, invalid: object,
) -> None:
    payload = _encode_persisted_domain(value)
    payload["fields"][field_name] = invalid
    with pytest.raises(AIError) as caught:
        _decode_domain(payload, type(value), _CURRENT_CODEC, persisted=True)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_recovery_defaults_still_enforce_cross_field_consistency() -> None:
    value = RecoveryCheckpoint(
        "execution", "run", RecoveryCheckpointState.COMPLETED, 1,
        datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 1, 1, tzinfo=timezone.utc),
        handoff_phase=RecoveryHandoffPhase.COMPLETED,
    )
    payload = _encode_persisted_domain(value)
    payload["fields"]["handoff_phase"]["value"] = RecoveryHandoffPhase.PREPARED.value
    with pytest.raises(AIError) as caught:
        _decode_domain(payload, RecoveryCheckpoint, _CURRENT_CODEC, persisted=True)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
