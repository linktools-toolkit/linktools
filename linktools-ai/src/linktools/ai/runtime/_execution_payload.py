#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Place frozen execution payloads only at their owning source boundary."""

from dataclasses import replace

from ..execution import ExecutionRecord
from ..inputs import ExecutionInputMaterializer


async def _materialize_execution_payloads(
    execution: ExecutionRecord, *, input_materializer: ExecutionInputMaterializer | None,
    tenant_id: str,
) -> ExecutionRecord:
    stored_input = execution.stored_user_input
    if stored_input is not None and input_materializer is not None:
        stored_input = await input_materializer.materialize_payload(stored_input, tenant_id=tenant_id)
    return replace(execution, stored_user_input=stored_input)
