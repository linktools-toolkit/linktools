#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Native decision models use the existing binding and durable observation APIs."""

from pathlib import Path
from typing import Literal

import pytest
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.usage import RequestUsage

pytest.importorskip("pydantic_ai.models.decision")
from pydantic_ai.models.decision import (
    ChoiceAnswer, DecisionModel, DecisionModelSettings, DecisionRequest,
    DecisionResponse, NoulAnswer, UnfillableRoute,
)

from linktools.ai.capability import TraceCapability
from linktools.ai.core import ExecutionStatus, JsonValue, Principal
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage


class _Decision(BaseModel):
    """Classify evidence without generating explanatory text."""

    accepted: bool
    category: Literal["accept", "reject"]
    probability: float = Field(ge=0, le=1)


class _OfflineDecisionModel(DecisionModel):
    def __init__(self, probability: float = 0.85) -> None:
        super().__init__()
        self.probability = probability
        self.requests: list[DecisionRequest] = []

    @property
    def model_name(self) -> str:
        return "offline-decision"

    @property
    def system(self) -> str:
        return "offline"

    async def decide(
        self, request: DecisionRequest, model_settings: DecisionModelSettings,
    ) -> DecisionResponse:
        self.requests.append(request)
        return DecisionResponse(
            answers={
                name: NoulAnswer(noul=self.probability)
                if question.type == "noul" else ChoiceAnswer(
                    choice="accept", confidence=0.8,
                    probabilities={"accept": 0.8, "reject": 0.2},
                )
                for name, question in request.questions.items()
            },
            model_name=self.model_name,
            usage=RequestUsage(input_tokens=12, output_tokens=3),
        )


class _Binding:
    provider = "offline"
    model_identity = "offline-decision"
    contract: dict[str, JsonValue] = {"provider": "offline", "model": "offline-decision"}

    def __init__(self, model: Model) -> None:
        self.model = model

    def materialize(self) -> Model:
        return self.model


def _features(models: list[_OfflineDecisionModel]) -> RuntimeFeatures:
    features = RuntimeFeatures("decisions")
    features.agent("default", allow_tools=())
    features.capability(TraceCapability(), scope="runtime")

    def load() -> _Binding:
        model = _OfflineDecisionModel(0.85 if not models else 0.15)
        models.append(model)
        return _Binding(model)

    features.model("default", load)
    return features


@pytest.mark.asyncio
async def test_native_decision_nonstream_uses_structured_output() -> None:
    model = _OfflineDecisionModel()
    result = await Agent(model, output_type=_Decision).run("Classify evidence")
    assert result.output == _Decision(accepted=True, category="accept", probability=0.85)
    assert len(model.requests) == 1
    assert result.usage.input_tokens == 12
    assert result.response.provider_details["probabilities"]["category"] == {
        "accept": 0.8, "reject": 0.2,
    }


@pytest.mark.asyncio
async def test_runtime_decision_stream_and_reopen_preserve_probabilities(tmp_path: Path) -> None:
    models: list[_OfflineDecisionModel] = []
    features = _features(models)
    database = tmp_path / "decisions.db"
    ids: list[str] = []
    principal = Principal("decision-reader", "default")
    async with Persistence.open('decisions', storage=PersistenceStorage.sqlite(database)) as persistence_110, Runtime.open(persistence=persistence_110, features=features) as runtime:
        for probability in (0.85, 0.15):
            execution = await runtime.agents.get().start("Classify evidence", output=_Decision, principal=principal)
            result = (await execution.wait(timeout_seconds=10)).result
            ids.append(execution.execution_id)
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {
                "accepted": probability >= 0.5, "category": "accept", "probability": probability,
            }
            assert result.usage.input_tokens == 12
            assert result.usage.output_tokens == 3
            assert result.usage.model_requests == 1
            page = await execution.model_interactions(include_content=True)
            assert len(page.items) == 1
            assert page.items[0].request["streaming"] is True
        assert len(models) == 2
        assert [len(model.requests) for model in models] == [1, 1]

    async with Persistence.open('decisions', storage=PersistenceStorage.sqlite(database)) as persistence_130, Runtime.open(persistence=persistence_130, features=features) as runtime:
        for execution_id in ids:
            page = await runtime.executions.model_interactions(execution_id, principal=principal, include_content=True)
            item = page.items[0]
            assert item.status == "SUCCEEDED"
            assert item.response["provider_details"]["probabilities"] == {
                "category": {"accept": 0.8, "reject": 0.2},
            }
            assert item.response["provider_details"]["confidence"] == {
                "accepted": 0.7, "category": 0.8,
            }
            assert item.usage.input_tokens == 12
            assert item.usage.output_tokens == 3
        assert len(models) == 2  # Historical reads never reload or call the model.


@pytest.mark.asyncio
async def test_runtime_does_not_retry_unhandled_decision_handoff() -> None:
    class Handoff(_OfflineDecisionModel):
        async def decide(self, request: DecisionRequest, model_settings: DecisionModelSettings) -> DecisionResponse:
            self.requests.append(request)
            raise UnfillableRoute(self.model_name, "review", 0.9)

    model = Handoff()
    features = RuntimeFeatures("unhandled")
    features.model("default", _Binding(model))
    features.agent("default", allow_tools=())
    async with Persistence.open('unhandled', storage=PersistenceStorage.in_memory()) as persistence_159, Runtime.open(persistence=persistence_159, features=features) as runtime:
        result = (await runtime.agents.get().run("Classify evidence", output=_Decision, timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_diagnostics.exception_type.endswith("UnfillableRoute")
    assert len(model.requests) == 1
