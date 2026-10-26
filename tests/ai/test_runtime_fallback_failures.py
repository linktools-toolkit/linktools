#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Exhausted fallback preserves billed attempts without turning them into requests."""

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic_ai import messages as sdk_messages
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from linktools.ai.capability import AgentContext, ReferencedInputsCapability, TraceCapability
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.model import FallbackBinding, ReferencedContent
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.spec import AgentUsageLimits

from .test_referenced_input_failure import _BODY, _REF, _SOURCE


pytestmark = pytest.mark.skipif(
    not hasattr(sdk_messages, "ModelRequestAttempt"),
    reason="installed SDK does not expose fallback attempt observations",
)


class _CompletedFallback(WrapperModel):
    """Adapt native nonstream response rejection to Runtime's streaming boundary."""

    @property
    def profile(self) -> ModelProfile:
        return ModelProfile()

    @asynccontextmanager
    async def request_stream(
        self, messages: list[ModelMessage], model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters, run_context: object | None = None,
    ) -> AsyncIterator[StreamedResponse]:
        del run_context
        response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        yield CompletedStreamedResponse(response, model_request_parameters=model_request_parameters)


class _Binding:
    provider = "test"
    model_identity = "test:all-failed"
    contract: Mapping[str, JsonValue] = {"input_mime_types": ["application/pdf"]}

    def __init__(self, model: Model) -> None:
        self.model = model

    def materialize(self) -> Model:
        return self.model


class _CostLimitCapability(AbstractCapability[AgentContext[None]]):
    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        assert ctx.usage_limits is not None
        ctx.usage_limits.cost_limit = Decimal("0.006")


@pytest.mark.asyncio
@pytest.mark.parametrize(("referenced", "limit"), [
    (False, None), (True, None), (False, "tokens"), (False, "cost"),
])
async def test_exhausted_fallback_retains_known_usage_and_safe_attempts_after_reopen(
    tmp_path: Path, referenced: bool, limit: str | None,
) -> None:
    requests: list[str] = []
    loaded: list[ReferencedContent] = []

    def response_model(name: str, input_tokens: int, output_tokens: int, cost: str) -> Model:
        def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            requests.append(name)
            return ModelResponse(
                [TextPart("rejected synthetic response")], model_name=name, provider_name="function",
                usage=RequestUsage(input_tokens=input_tokens, output_tokens=output_tokens, cost=Decimal(cost)),
            )
        return FunctionModel(respond, model_name=name)

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        requests.append("unavailable")
        raise ModelAPIError("unavailable", 'password="failure-secret" detail=unavailable')

    def reject(response: ModelResponse) -> bool:
        del response
        return True

    def features() -> RuntimeFeatures:
        group = RuntimeFeatures("exhausted-fallback")
        group.agent(
            "default", allow_tools=(), allow_skills=(), allow_subagents=(),
            usage_limits=AgentUsageLimits(input_tokens=10) if limit == "tokens" else None,
        )
        group.capability(TraceCapability(record_http_bodies=True), scope="runtime")
        if limit == "cost":
            group.capability(_CostLimitCapability(), id="cost-budget", scope="runtime")
        native = FallbackModel(
            response_model("first", 7, 3, "0.003"),
            response_model("second", 11, 5, "0.005"),
            FunctionModel(fail, model_name="unavailable"),
            fallback_on=[reject, ModelAPIError],
        )
        group.model("default", FallbackBinding(
            _Binding(_CompletedFallback(native)), fallback_on=FallbackExceptionGroup,
        ))
        if referenced:
            async def load(context: AgentContext[None], content: ReferencedContent) -> bytes:
                del context
                loaded.append(content)
                return _BODY
            group.capability(ReferencedInputsCapability(load), id=_SOURCE.id,
                             revision=_SOURCE.revision, scope="agent")
        return group

    prompt = ["inspect", ReferencedContent(_REF, media_type="application/pdf", source=_SOURCE)] if referenced else "inspect"
    async with Persistence.open('exhausted-fallback', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_131, Runtime.attach(persistence=persistence_131, features=features()) as runtime:
        execution = await runtime.agents.get().start(prompt)
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ("INTERNAL_ERROR" if limit is None else "EXECUTION_USAGE_LIMIT_EXCEEDED")
        assert result.usage.input_tokens == 18
        assert result.usage.output_tokens == 8
        assert result.usage.model_requests == 0
        attempts = result.safe_error_details["failed_attempts"]
        assert [item["outcome"] for item in attempts] == ["rejected", "rejected", "error", "error"]
        assert [item["model_name"] for item in attempts[:3]] == ["first", "second", "unavailable"]
        assert [item["usage"]["input_tokens"] for item in attempts[:2]] == [7, 11]
        assert [item["usage"]["cost"] for item in attempts[:2]] == ["0.003", "0.005"]
        assert all(item["usage"] is None for item in attempts[2:])
        assert all(isinstance(item["duration_us"], int) and item["duration_us"] >= 0 for item in attempts)
        assert "failure-secret" not in repr(result.safe_error_details)
        assert "failure-secret" not in repr(result.error_diagnostics)
        if referenced:
            assert all(item["error"] is None for item in attempts)
            assert result.error_diagnostics.content_omission_reason == "referenced_content_lifecycle"
        else:
            assert "detail=unavailable" in attempts[2]["error"]
        interactions = (await execution.model_interactions(include_content=True)).items
        assert len(interactions) == 1
        assert interactions[0].status == "FAILED"
        assert interactions[0].response is None
        assert interactions[0].error_code == result.error_code
        assert interactions[0].usage.input_tokens == 18
        assert interactions[0].usage.output_tokens == 8
        assert interactions[0].usage.model_requests == 1
        execution_id, principal = execution.execution_id, runtime.default_principal

    async with Persistence.open('exhausted-fallback', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_165, Runtime.attach(persistence=persistence_165, features=features()) as runtime:
        restored = await runtime.executions.result(execution_id, principal=principal)
        assert restored.safe_error_details == result.safe_error_details
        assert restored.usage == result.usage
        assert restored.error_code == result.error_code
        restored_interactions = (await runtime.executions.model_interactions(
            execution_id, principal=principal, include_content=True,
        )).items
        assert restored_interactions[0].usage == interactions[0].usage
    assert requests == ["first", "second", "unavailable"]
    assert len(loaded) == (1 if referenced else 0)


@pytest.mark.asyncio
async def test_exhausted_provider_errors_keep_model_api_classification_and_unknown_usage() -> None:
    requested: list[str] = []

    def candidate(name: str) -> Model:
        async def fail(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            del messages, info
            requested.append(name)
            raise ModelAPIError(name, 'password="provider-secret" detail=unavailable')
            yield ""  # pragma: no cover
        return FunctionModel(stream_function=fail, model_name=name)

    features = RuntimeFeatures("provider-failures")
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    features.model("default", FallbackBinding(_Binding(candidate("first")), _Binding(candidate("second"))))
    features.capability(TraceCapability(), scope="runtime")
    async with Persistence.open('provider-failures', storage=PersistenceStorage.in_memory()) as persistence_196, Runtime.attach(persistence=persistence_196, features=features) as runtime:
        execution = await runtime.agents.get().start("inspect")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == "MODEL_API_ERROR"
        attempts = result.safe_error_details["failed_attempts"]
        assert [item["model_name"] for item in attempts] == ["first", "second"]
        assert all(item["outcome"] == "error" and item["usage"] is None for item in attempts)
        assert "provider-secret" not in repr(result.safe_error_details)
        assert "provider-secret" not in repr(result.error_diagnostics)
        assert "detail=unavailable" in attempts[0]["error"]
        assert result.usage.input_tokens == result.usage.output_tokens == result.usage.model_requests == 0
        interactions = (await execution.model_interactions(include_content=True)).items
        assert len(interactions) == 1
        assert interactions[0].status == "FAILED"
        assert interactions[0].error_code == "MODEL_API_ERROR"
        assert interactions[0].response is None
        assert interactions[0].usage is None
    assert requested == ["first", "second"]
