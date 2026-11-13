#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model coverage validates its binding without restoring result payloads."""

from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence.contracts import ExecutionObservationRecord

from .test_model_capture_reads import _NAMESPACE, _PRINCIPAL, _seed, _service
from .test_incremental_projection_boundaries import _storage


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_observation_does_not_expand_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    state = Persistence(_storage(backend, tmp_path))
    await state.initialize(namespace=_NAMESPACE, tenant_id="tenant")
    try:
        await _seed(state)
        repository = state.execution.executions
        async def forbidden(*args, **kwargs):
            raise AssertionError("model coverage must not restore result payloads")
        monkeypatch.setattr(repository._execution_content, "expand_projection_components", forbidden)
        observation = await repository.get_observation("execution", tenant_id="tenant")
        assert isinstance(observation, ExecutionObservationRecord)
        assert observation.execution_id == "execution" and observation.agent_run_seq == 1
        assert await repository.get_observation("execution", tenant_id="other") is None
        service = _service(state)
        boundary, items = await service.read_model_interaction_updates("execution", principal=_PRINCIPAL)
        assert boundary.cutoffs[0].model_request_seq == 1
        assert len(items) == 1 and items[0].model_request_seq == 1
        class Deny:
            async def authorize(self, principal, action, resource):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        with pytest.raises(AIError) as error:
            await _service(state, authorization=Deny()).read_model_interaction_updates("execution", principal=_PRINCIPAL)
        assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()
