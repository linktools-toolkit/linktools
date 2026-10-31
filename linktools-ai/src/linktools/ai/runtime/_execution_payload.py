#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Place frozen execution payloads only at their owning source boundary."""

from dataclasses import replace

from ..core import PersistenceDomain, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..execution import ExecutionRecord
from ..inputs import ExecutionInputMaterializer
from ..persistence.objects import ObjectKeyFactory, put_runtime_object
from ..storage import ObjectStore, PayloadPolicy, StoredPayload, payload_fits_inline


async def _materialize_execution_payloads(
    execution: ExecutionRecord, *, input_materializer: ExecutionInputMaterializer | None,
    objects: ObjectStore, keys: ObjectKeyFactory | None, policy: PayloadPolicy | None, tenant_id: str,
) -> ExecutionRecord:
    stored_input = execution.stored_user_input
    if stored_input is not None and input_materializer is not None:
        stored_input = await input_materializer.materialize_payload(stored_input, tenant_id=tenant_id)
    instructions = execution.repository_instructions
    if (instructions is not None and instructions.payload.kind == "inline"
            and instructions.source_domain is PersistenceDomain.EXECUTION):
        if keys is None or policy is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        instructions = replace(instructions, payload=await _materialize_execution_json(
            instructions.payload, objects=objects, keys=keys, policy=policy, tenant_id=tenant_id,
        ))
    return replace(execution, stored_user_input=stored_input, repository_instructions=instructions)


async def _materialize_execution_json(
    payload: StoredPayload, *, objects: ObjectStore, keys: ObjectKeyFactory,
    policy: PayloadPolicy, tenant_id: str,
) -> StoredPayload:
    if payload.kind == "object" or payload_fits_inline(payload, policy):
        return payload
    if payload.encoding != "json":
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    reference = await put_runtime_object(
        objects, keys, PersistenceDomain.EXECUTION, tenant_id, canonical_json_bytes(payload.value),
    )
    return StoredPayload.object(reference)
