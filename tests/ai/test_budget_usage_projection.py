#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordinary budget queries consume authority and scope, not recovery bodies."""

from contextvars import ContextVar
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus, Principal, RunBudget
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._execution_content import ExecutionContent
from linktools.ai.runtime import Runtime

from ._runtime_test_helpers import RuntimeUsageModels, _UsageFunctionModel, model_features
from .test_run_budget_public_api import _capabilities


class _LargeModels(RuntimeUsageModels):
    def materialize(self) -> _UsageFunctionModel:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[TextPart("large result " * 10000)])
        return _UsageFunctionModel(respond)


@pytest.mark.asyncio
async def test_budget_usage_view_skips_bodies_and_preserves_pending_and_custom_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    features = _capabilities()
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open("budget-view", storage=PersistenceStorage.in_memory()) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_LargeModels(), features)) as runtime:
            handle = await runtime.agents.get().start("large accepted input " * 10000, budget=RunBudget(model_requests=10))
            assert (await handle.wait()).result.status is ExecutionStatus.SUCCEEDED
            execution_id = handle.execution_id
        async with Runtime.attach(persistence=persistence, features=model_features(_LargeModels(), features)) as runtime:
            handle = await runtime.executions.get(execution_id, principal=runtime.default_principal)
            repository = persistence.execution.executions
            original = await repository.get_fresh(handle.execution_id, tenant_id="default")
            assert original is not None and original.budget_scope_id is not None and original.result is not None
            expected = await persistence.execution.budgets.read(original.budget_scope_id)
            service = runtime._execution_service

            async def forbid(*args: object, **kwargs: object) -> None:
                raise AssertionError("budget usage must not hydrate recovery input or output")

            query_active = ContextVar("budget_query_active", default=False)
            read_payloads = ExecutionContent.read_payloads
            async def guarded_read(self: ExecutionContent, *args: object, **kwargs: object) -> object:
                if query_active.get():
                    return await forbid()
                return await read_payloads(self, *args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(ExecutionContent, "read_payloads", guarded_read)
                query_active.set(True)
                patch.setattr(service, "load_observed_authorized", forbid)
                assert await handle.budget_usage() == expected
                with pytest.raises(AIError) as denied:
                    await runtime.executions.budget_usage(handle.execution_id, principal=Principal("other", "other"))
                assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                with pytest.raises(AIError) as missing:
                    await runtime.executions.budget_usage("missing", principal=runtime.default_principal)
                assert missing.value.code is ErrorCode.AUTHORIZATION_DENIED

                pending = replace(original, budget_scope_id="pending-budget")
                await persistence.execution.budgets.ensure(pending.budget_scope_id, RunBudget(model_requests=7))

                async def read_pending(*args: object, **kwargs: object) -> object:
                    return pending

                patch.setattr(service._runtime_bridge.pending_turns, "read_execution", read_pending)
                usage = await handle.budget_usage()
                assert usage is not None and usage.scope_id == pending.budget_scope_id
                pending = replace(pending, budget_scope_id=None)
                assert await handle.budget_usage() is None

            query_active.set(False)
            calls = []
            async def custom_read(*args: object, **kwargs: object) -> object:
                calls.append(args)
                return original

            with monkeypatch.context() as patch:
                patch.setattr(service, "_state", SimpleNamespace(executions=object()))
                patch.setattr(service, "load_observed_authorized", custom_read)
                assert await service.budget_usage(handle.execution_id, principal=runtime.default_principal) == expected
                assert len(calls) == 1
