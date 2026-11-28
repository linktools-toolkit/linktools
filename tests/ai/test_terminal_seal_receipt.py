#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Only matching, atomically confirmed terminal seals waive cleanup reads."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import TraceCapability
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistencePlan, PersistenceStorage
from linktools.ai.persistence._execution_repositories import ExecutionRepositoryImpl
from linktools.ai.persistence._step_archive import StateStepArchive
from linktools.ai.persistence.contracts import ExecutionTerminalPublication
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


@pytest.mark.asyncio
@pytest.mark.parametrize("proof", (
    "confirmed", "missing", "wrong_execution", "wrong_digest",
    "wrong_generation", "lost_ack", "separate_conversation", "split_trace",
))
async def test_terminal_cleanup_requires_confirmed_matching_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proof: str,
) -> None:
    route = StorageRoute.sqlite(tmp_path / "execution.db")
    storage = PersistenceStorage.from_plan(PersistencePlan(
        execution=route,
        trace=StorageRoute.sqlite(tmp_path / "trace.db") if proof == "split_trace" else route,
        conversation=(StorageRoute.sqlite(tmp_path / "conversation.db")
                      if proof == "separate_conversation" else route),
    ))
    features = RuntimeFeatures("terminal-seal-receipt", defaults={"execution": {"retain_history": True}})
    features.agent("default", model="default", allow_tools=())
    features.capability(TraceCapability(), scope="runtime")
    publications, checks, provider_calls = [], [], []
    original_revoke = StateStepArchive.require_revoked_execution_producer
    original_check = ExecutionRepositoryImpl.require_revoked_producer

    async def check(self, *args, **kwargs):
        checks.append(args)
        return await original_check(self, *args, **kwargs)

    async def revoke(self, **kwargs):
        publication = kwargs.get("terminal_publication")
        publications.append(publication)
        if proof == "lost_ack":
            assert publication is None
            return await original_revoke(self, **kwargs)
        assert publication is not None
        seal = publication.history_seal
        assert seal is not None
        assert seal.execution_id == publication.execution.execution_id
        assert seal.seal_digest == publication.execution.history_seal_digest
        if proof == "missing":
            publication = replace(publication, history_seal=None)
        elif proof == "wrong_execution":
            publication = replace(publication, history_seal=replace(seal, execution_id="other"))
        elif proof == "wrong_digest":
            changed = replace(seal, execution_event_high_water=seal.execution_event_high_water + 1)
            assert changed.execution_id == publication.execution.execution_id
            assert changed.seal_digest != publication.execution.history_seal_digest
            publication = replace(publication, history_seal=changed)
        elif proof == "wrong_generation":
            publication = replace(publication, execution=replace(
                publication.execution,
                producer_generation=publication.execution.producer_generation + 1,
            ))
        kwargs["terminal_publication"] = publication
        return await original_revoke(self, **kwargs)

    monkeypatch.setattr(StateStepArchive, "require_revoked_execution_producer", revoke)
    monkeypatch.setattr(ExecutionRepositoryImpl, "require_revoked_producer", check)

    def respond(_messages, _info):
        provider_calls.append(True)
        return ModelResponse(parts=[TextPart("complete")])

    async with Persistence.open("terminal-seal-receipt", storage=storage) as state, Runtime.open(
        persistence=state,
        features=model_features(_Models(_UsageFunctionModel(respond)), features),
    ) as runtime:
        lost_acknowledgements = []
        if proof == "lost_ack":
            group_type = type(state.execution.executions.state_store.storage_group)
            original_mutate = group_type.mutate

            async def lose_ack(group, *args, **kwargs):
                value = await original_mutate(group, *args, **kwargs)
                if isinstance(value, ExecutionTerminalPublication):
                    # The actual transaction committed, but its receipt was lost.
                    lost_acknowledgements.append(value)
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return value

            monkeypatch.setattr(group_type, "mutate", lose_ack)

        # Seals belong to independent retained Executions. Native Session
        # trace is an optional observer and cannot authorize producer cleanup.
        agent = runtime.agents.get()
        if proof == "split_trace":
            with pytest.raises(AIError) as unsupported:
                await agent.start("hello", idempotency_key="turn-0")
            assert unsupported.value.code is ErrorCode.STORAGE_CAPABILITY_MISSING
            assert provider_calls == [] and publications == [] and checks == []
            return

        first_ids = []
        for turn in range(2):
            run = await agent.start("hello", idempotency_key=f"turn-{turn}")
            assert (await run.wait(timeout_seconds=20)).result.output == {"text": "complete"}
            first_ids.append(run.execution_id)
        repeated = await agent.start("hello", idempotency_key="turn-0")
        assert repeated.execution_id == first_ids[0]
        assert (await repeated.wait(timeout_seconds=20)).result.output == {"text": "complete"}
        assert provider_calls == [True, True]

        assert len(publications) == 2
        assert len(checks) == (0 if proof in {"confirmed", "separate_conversation"} else 2)
        assert len(lost_acknowledgements) == (2 if proof == "lost_ack" else 0)
