#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure deterministic identities for Runtime-owned observations."""

from ..core import canonical_sha256


def stable_observation_id(*parts: str) -> str:
    return canonical_sha256({"parts": list(parts)})


def model_observation_id(
    source_namespace: str,
    tenant_id: str,
    execution_id: str,
    agent_run_id: str,
    model_request_seq: int,
    purpose: str,
) -> str:
    return stable_observation_id(
        "linktools.model.request.v1",
        source_namespace,
        tenant_id,
        execution_id,
        agent_run_id,
        str(model_request_seq),
        purpose,
    )


def tool_observation_id(
    source_namespace: str,
    tenant_id: str,
    execution_id: str,
    agent_run_id: str,
    tool_call_id: str,
) -> str:
    return stable_observation_id(
        "linktools.tool.execution.v1",
        source_namespace,
        tenant_id,
        execution_id,
        agent_run_id,
        tool_call_id,
    )


__all__ = ["stable_observation_id", "model_observation_id", "tool_observation_id"]
