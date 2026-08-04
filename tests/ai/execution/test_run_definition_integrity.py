#!/usr/bin/env python3
"""Run definitions are content-addressed by their canonical schema and spec."""

import pytest

from linktools.ai.errors import RunDefinitionIntegrityError
from linktools.ai.execution.domain import (
    RunDefinition,
    RunnableType,
    compute_run_definition_hash,
)


def test_definition_hash_is_canonical_across_mapping_order() -> None:
    first = {"b": 2, "a": {"y": 1, "x": 0}}
    second = {"a": {"x": 0, "y": 1}, "b": 2}
    assert compute_run_definition_hash(schema="agent-spec.v1", spec=first) == (
        compute_run_definition_hash(schema="agent-spec.v1", spec=second)
    )


def test_definition_hash_includes_schema() -> None:
    spec = {"id": "agent"}
    assert compute_run_definition_hash(schema="agent-spec.v1", spec=spec) != (
        compute_run_definition_hash(schema="swarm-spec.v1", spec=spec)
    )


def test_definition_rejects_spec_or_schema_tampering() -> None:
    spec = {"id": "agent"}
    digest = compute_run_definition_hash(schema="agent-spec.v1", spec=spec)
    with pytest.raises(RunDefinitionIntegrityError):
        RunDefinition(
            "agent",
            RunnableType.AGENT,
            "agent-spec.v1",
            {"id": "changed"},
            digest,
        )
    with pytest.raises(RunDefinitionIntegrityError):
        RunDefinition("agent", RunnableType.AGENT, "swarm-spec.v1", spec, digest)
