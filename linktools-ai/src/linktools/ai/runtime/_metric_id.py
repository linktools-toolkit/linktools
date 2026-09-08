#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure deterministic identities for Runtime-owned observations."""

from ..core import canonical_sha256


def _stable_observation_id(*parts: str) -> str:
    return canonical_sha256({"parts": list(parts)})


def _model_observation_id(
    source_namespace: str,
    tenant_id: str,
    execution_id: str,
    step_run_id: str,
    request_sequence: int,
    purpose: str,
) -> str:
    return _stable_observation_id(
        "linktools.model.request.v2",
        source_namespace,
        tenant_id,
        execution_id,
        step_run_id,
        str(request_sequence),
        purpose,
    )


def _tool_observation_id(
    source_namespace: str,
    tenant_id: str,
    execution_id: str,
    step_run_id: str,
    tool_call_id: str,
) -> str:
    return _stable_observation_id(
        "linktools.tool.execution.v1",
        source_namespace,
        tenant_id,
        execution_id,
        step_run_id,
        tool_call_id,
    )


__all__: list[str] = []
