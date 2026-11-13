#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical tool effect identity, carried independently of provider call IDs."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..persistence.runs import StepEvent
from ..errors import AIError, ErrorCode

from ..core import canonical_sha256, validate_resource_id, validate_tenant_id

TOOL_OPERATION_ID_METADATA_KEY = "linktools.operation_id"


def tool_operation_id(*, tenant_id: str, execution_id: str, agent_run_id: str, tool_call_id: str) -> str:
    """Allocate once from the original admission, never from a replay run."""
    return canonical_sha256({
        "tenant_id": validate_tenant_id(tenant_id),
        "execution_id": validate_resource_id(execution_id),
        "agent_run_id": validate_resource_id(agent_run_id),
        "tool_call_id": validate_resource_id(tool_call_id),
    })


def tool_operation_references(events: Sequence["StepEvent"]) -> dict[str, str]:
    """Read admission-owned identities from immutable Step facts."""
    result = {}
    for event in events:
        identity = event.metadata.get(TOOL_OPERATION_ID_METADATA_KEY)
        if identity is None:
            continue
        call_id = event.tool_call_id
        if not isinstance(call_id, str) or not isinstance(identity, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            validate_resource_id(call_id)
            validate_resource_id(identity)
        except AIError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        if call_id in result and result[call_id] != identity:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result[call_id] = identity
    return result
