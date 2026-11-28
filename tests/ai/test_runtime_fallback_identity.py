#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Successful fallback identity survives Runtime publication and SQLite reopening."""

import asyncio

import pytest
from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, ModelFallback
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import SessionCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures

from .test_decision_model_binding import _Binding


class _SessionAuthorization(TenantAuthorizationPolicy):
    """The test fixture owns every session in its isolated namespace."""


@pytest.mark.asyncio
@pytest.mark.parametrize("checkpoints", (False, True))
@pytest.mark.parametrize("selection", ("primary", "secondary", "alias", "producer", "replaced", "compaction_replaced"))
async def test_successful_model_identity_survives_sqlite_reopen(tmp_path, selection, checkpoints):
    def model(name, fail=False):
        async def respond(messages, info):
            if fail:
                raise ModelAPIError(name, "synthetic failure")
            return ModelResponse([TextPart("answer")], model_name=name)

        async def stream(messages, info):
            if fail:
                raise ModelAPIError(name, "synthetic failure")
            yield "answer"

        return FunctionModel(
            respond, stream_function=stream, model_name=name,
            profile={"context_window": 1000} if selection == "compaction_replaced" else None,
        )

    def models_and_capabilities():
        first = model("first", fail=selection != "primary")
        second = model("second")

        class Identity(AbstractCapability):
            async def resolve_model_id(self, ctx, *, model_id):
                return second if model_id == "serving-alias" else None

            async def after_model_request(self, ctx, *, request_context, response):
                if selection in {"replaced", "compaction_replaced"}:
                    return ModelResponse(response.parts, model_name=response.model_name)
                if selection == "producer":
                    response.model_id = "producer-identity"
                return response

        return first, [ModelFallback("serving-alias" if selection == "alias" else second), Identity()]

    expected_name = "first" if selection == "primary" else "second"
    expected_id = {"alias": "serving-alias", "producer": "producer-identity"}.get(
        selection, f"function:{expected_name}",
    )
    first, capabilities = models_and_capabilities()
    native = await Agent(first, capabilities=capabilities).run("hello")
    assert (native.response.model_name, native.response.model_id) == (
        expected_name, expected_id,
    )

    def features():
        first, capabilities = models_and_capabilities()
        result = RuntimeFeatures("fallback-identity", defaults={"execution": {"retain_history": True}})
        result.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
        result.model("default", _Binding(first))
        for index, capability in enumerate(capabilities):
            result.capability(capability, id=f"identity-{index}")
        result.capability(SessionCapability(checkpoints=checkpoints))
        return result

    async def assert_persisted(persistence):
        turns = await persistence.session_turn_reader.read_context_range("session", after=0, through=1)
        responses = [message for turn in turns for message in turn.messages if isinstance(message, ModelResponse)]
        assert [(message.model_name, message.model_id) for message in responses] == [(expected_name, expected_id)]

    storage = PersistenceStorage.sqlite(tmp_path / "state.sqlite")
    async with Persistence.open("fallback-identity", storage=storage) as persistence, Runtime.open(
        persistence=persistence, features=features(), authorization=_SessionAuthorization(),
    ) as runtime:
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        turn = await runtime.sessions.start_turn(
            "session", "hello", agent="default", principal=principal, idempotency_key="start",
        )
        assert (await turn.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        await assert_persisted(persistence)

    async with Persistence.open("fallback-identity", storage=storage) as persistence:
        await assert_persisted(persistence)


@pytest.mark.asyncio
async def test_parallel_fallback_and_primary_keep_distinct_response_identities(tmp_path):
    entered = {"fallback": asyncio.Event(), "primary": asyncio.Event()}

    def model(name):
        async def stream(messages, info):
            if name == "failed":
                raise ModelAPIError(name, "synthetic failure")
            entered[name].set()
            await asyncio.wait_for(entered["primary" if name == "fallback" else "fallback"].wait(), 5)
            yield name

        return FunctionModel(stream_function=stream, model_name=name)

    features = RuntimeFeatures("parallel-identities", defaults={"execution": {"retain_history": True}})
    for agent, selected in (("recovering", "failed"), ("direct", "primary")):
        features.agent(agent, model=agent, allow_tools=(), allow_skills=(), allow_subagents=())
        features.model(agent, _Binding(model(selected)))
    features.capability(ModelFallback(model("fallback")), id="fallback")
    storage = PersistenceStorage.sqlite(tmp_path / "parallel.sqlite")
    async with Persistence.open("parallel-identities", storage=storage) as persistence, Runtime.open(
        persistence=persistence, features=features, authorization=_SessionAuthorization(),
    ) as runtime:
        principal = runtime.default_principal

        async def run(agent, expected):
            await runtime.sessions.create_turn_session(agent, principal=principal)
            turn = await runtime.sessions.start_turn(
                agent, "hello", agent=agent, principal=principal, idempotency_key="start",
            )
            assert (await turn.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
            turns = await persistence.session_turn_reader.read_context_range(agent, after=0, through=1)
            responses = [message for entry in turns for message in entry.messages if isinstance(message, ModelResponse)]
            assert [(message.model_name, message.model_id) for message in responses] == [
                (expected, f"function:{expected}"),
            ]
            return responses[0].run_id

        run_ids = await asyncio.gather(run("recovering", "fallback"), run("direct", "primary"))
        assert all(run_ids) and len(set(run_ids)) == 2
