#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused regression coverage for the v3 Agent composition contract."""

from types import SimpleNamespace

import pytest
from pydantic_ai.capabilities import AbstractCapability

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.capability import RuntimeCapability
from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, SessionView
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import trusted_workspace_principal


class _DurableCapability(AbstractCapability[None]):
    @classmethod
    def get_serialization_name(cls) -> "str | None":
        return "test-durable-capability"

    @classmethod
    def from_spec(cls, **kwargs: object) -> "_DurableCapability":
        del kwargs
        return cls()


def test_runtime_capability_from_spec_requires_exact_importable_type() -> None:
    shadow_type = type(
        "_DurableCapability",
        (_DurableCapability,),
        {
            "__module__": __name__,
            "__qualname__": "_DurableCapability",
        },
    )

    with pytest.raises(AIError) as error:
        RuntimeCapability.from_spec("local", shadow_type, config={})

    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


def test_agent_binding_snapshot_is_deeply_immutable() -> None:
    snapshot = AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "model"),
        output_type_module="example.output",
        output_type_qualname="Output",
        output_schema_id="output",
        output_schema_revision=1,
        output_schema_fingerprint="b" * 64,
        local_runtime_capability_descriptors=(
            {"config": {"items": ["first"]}},
        ),
        binding_digest="a" * 64,
    )

    descriptor = snapshot.local_runtime_capability_descriptors[0]
    exposed_config = descriptor["config"]
    assert isinstance(exposed_config, dict)
    exposed_items = exposed_config["items"]
    assert isinstance(exposed_items, list)
    exposed_items.append("mutated")

    persisted = snapshot.to_payload()["local_runtime_capability_descriptors"]
    assert persisted == [{"config": {"items": ["first"]}}]


class _SessionService:
    def __init__(self, binding_digest: str) -> None:
        self._binding_digest = binding_digest

    async def get(self, session_id: str, *, principal: object) -> SessionView:
        del principal
        return SessionView(
            session_id,
            self._binding_digest,
            SessionStatus.OPEN,
        )


@pytest.mark.asyncio
async def test_runtime_session_definition_reports_definition_conflict() -> None:
    runtime = object.__new__(Runtime)
    runtime.session = _SessionService("b" * 64)
    runtime._catalog = SimpleNamespace(definition=lambda digest: digest)
    preferred = SimpleNamespace(digest="a" * 64)

    with pytest.raises(AIError) as error:
        await runtime._session_definition(
            "session",
            trusted_workspace_principal("tenant"),
            preferred=preferred,
        )

    assert error.value.code is ErrorCode.SESSION_CONFLICT


@pytest.mark.asyncio
async def test_runtime_existing_session_reports_definition_conflict() -> None:
    runtime = object.__new__(Runtime)
    runtime.session = _SessionService("b" * 64)
    definition = SimpleNamespace(
        digest="a" * 64,
        spec=SimpleNamespace(id="agent"),
    )

    with pytest.raises(AIError) as error:
        await runtime._ensure_session(
            definition,
            "session",
            trusted_workspace_principal("tenant"),
        )

    assert error.value.code is ErrorCode.SESSION_CONFLICT
