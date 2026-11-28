#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Independent regression for accepted-context archival at effect promotion."""
from ._runtime_test_helpers import model_features
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, UserPromptPart

from linktools.ai.capability import AgentContext
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import CommitPolicy, ExecutionStatus, TenantAuthorizationPolicy, canonical_json_bytes
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.storage._database import SqlStorageContext
from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize('effect_policy', ('none', 'replay_safe', 'non_replay_safe'))
async def test_promotion_archives_accepted_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effect_policy: str,
) -> None:
    import asyncio
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    calls = []
    state = PersistenceStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024), coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    owner = {}
    ready = asyncio.Event()
    async def lookup(context: AgentContext[None]) -> str:
        record = await runtime._state.execution.executions.get(context.execution_id, tenant_id=context.principal.tenant_id)
        if record is not None:
            assert record.input_context.payload.decode()['kind'] == 'execution-context'
        calls.append(record)
        return 'value'
    async def model(messages, info):
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            await ready.wait()
            assert await runtime._state.execution.executions.get(owner['execution'], tenant_id=owner['tenant']) is None
            return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='lookup-call')])
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('tool-pending', defaults={"execution": {"retain_history": True}}, execution_commit=CommitPolicy.terminal())
    group.tool(lookup, effect_policy=effect_policy)
    group.agent('default', model='default', allow_tools=('lookup',))
    async with Persistence.open('pending-tool', storage=state) as persistence_43, Runtime.open(persistence=persistence_43, features=model_features(_Models(_UsageFunctionModel(model)), group)) as runtime:
        agent = runtime.agents.get()
        original = SqlStorageContext.run_mutation
        commits = []
        cleanups = []
        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            domain = kwargs.get("domain")
            (cleanups if domain == "native.content_cleanup" else commits).append(domain)
            return result
        monkeypatch.setattr(SqlStorageContext, 'run_mutation', count)
        execution = await agent.start('question')
        owner.update(execution=execution.execution_id, tenant=runtime.tenant_id)
        ready.set()
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert len(calls) == 1
        if effect_policy != 'non_replay_safe':
            assert calls == [None]
            assert len(commits) == 1
            assert len(cleanups) <= 1
        else:
            assert calls[0] is not None and calls[0].status is ExecutionStatus.STARTED
            assert len(commits) > 1
            assert len(cleanups) <= len(commits)


@pytest.mark.asyncio
async def test_native_session_admission_reuses_verified_turn_history(tmp_path: Path) -> None:
    class HostAuthorization(TenantAuthorizationPolicy):
        """Provide session authority for the SQL runtime."""

    prompts = ("first", "second", "third")
    seen = []

    async def model(messages, _info):
        visible = [part.content for message in messages for part in message.parts
                   if isinstance(part, UserPromptPart)]
        assert visible == list(prompts[:len(seen) + 1])
        seen.append(visible[-1])
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("review-context", defaults={"execution": {"retain_history": True}})
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.from_root(tmp_path / "context")
    async with (
        Persistence.open("review-context", storage=storage) as state,
        Runtime.open(persistence=state, authorization=HostAuthorization(),
                     features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime,
    ):
        agent = runtime.agents.get()
        await agent.create_session("session")
        snapshots = {}
        for prompt in prompts:
            turn = await agent.session("session").start(prompt, idempotency_key=prompt)
            assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
            row = await state.session_turns.get_turn(turn.key)
            snapshots[turn.key] = canonical_json_bytes(dict(row)["result_json"])
        for key, result in snapshots.items():
            row = await state.session_turns.get_turn(key)
            assert canonical_json_bytes(dict(row)["result_json"]) == result
            assert await state.execution.executions.get(key.turn_id, tenant_id=runtime.tenant_id) is None
        assert seen == list(prompts)


@pytest.mark.asyncio
async def test_session_admission_captures_current_rules_without_rewriting_history(tmp_path: Path) -> None:
    from linktools.ai.asset import AssetKey, AssetStore, InMemoryAssetBackend
    from linktools.ai.capability import AssetCapability
    from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions
    from linktools.ai.storage import StorageOverlay

    class HostAuthorization(TenantAuthorizationPolicy):
        """Supply session authority without binding session to an Agent."""

    backend = InMemoryAssetBackend()
    assets = AssetStore(StorageOverlay(backend, writer=backend))
    await assets.initialize()
    observed = []

    async def model(messages, info):
        observed.append(info.instructions or "")
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("current-instructions", defaults={"execution": {"retain_history": True}})
    features.capability(AssetCapability(assets))
    features.agent("default", model="default", allow_tools=())
    rule = AssetKey("rule", "current.md")
    expected = None
    try:
        async with Persistence.open("current-instructions", storage=PersistenceStorage.from_root(tmp_path / "rules")) as persistence, Runtime.open(
            persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
            authorization=HostAuthorization(),
        ) as runtime:
            agent = runtime.agents.get()
            await agent.create_session("session")
            records = []
            for index in range(3):
                if index == 1:
                    await assets.put(rule, b"Rule added after startup")
                    accepted_version, = await assets.resolve_versions((rule,))
                    expected = RepositoryInstructions((RepositoryInstructionDocument(
                        "rule:current", ".", "Rule added after startup",
                        asset_source_id="current-instructions", asset=accepted_version,
                    ),))
                elif index == 2:
                    await assets.delete(rule)
                turn = await agent.session("session").start(f"question {index}",
                                         idempotency_key=f"question-{index}")
                assert (await turn.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
                row = await persistence.session_turns.get_turn(turn.key)
                records.append((turn.key, row["request_json"]))
            empty = RepositoryInstructions(()).to_payload()
            assert records[0][1]["repository_instructions"] == empty
            assert expected is not None
            assert records[1][1]["repository_instructions"] == expected.to_payload()
            assert records[2][1]["repository_instructions"] == empty
            for key, request in records:
                assert (await persistence.session_turns.get_turn(key))["request_json"] == request
            assert "Rule added after startup" not in observed[0]
            assert "Rule added after startup" in observed[1]
            assert "Rule added after startup" not in observed[2]
    finally:
        await assets.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('rounds', (2, 4))
async def test_projection_boundary_read_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rounds: int) -> None:
    import json
    from contextvars import ContextVar
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    from linktools.ai.capability import TraceCapability
    from pydantic_ai_harness.step_persistence import StepPersistence
    from tests.ai._runtime_test_helpers import native_checkpoint
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.persistence._history import TranscriptRepository
    phase = ContextVar('review_phase', default='other')
    observed = []
    owner_reads = []
    details = []
    original_read = ContentArchive._read_content
    async def read(self, content):
        if content.codec == "model-message":
            observed.append((phase.get(), content.identity, content.raw_size, "body" if content.payload is None else content.payload.kind))
        else:
            from linktools.ai.persistence.state import active_state_transaction
            assert content.codec in {"execution-binding", "runtime-policy", "execution-payload"}
            owner_reads.append((active_state_transaction(self.store), content.identity, content.raw_size))
        return await original_read(self, content)
    def wrap(method, label):
        async def invoke(self, *args, **kwargs):
            token = phase.set(label)
            before = len(observed)
            try:
                return await method(self, *args, **kwargs)
            finally:
                if label == 'projection':
                    baseline = kwargs.get('baseline')
                    root = getattr(baseline.items[0], 'root', None) if baseline is not None and baseline.items else None
                    details.append({'domain':self._runtime_domain.value, 'messages':len(args[0]), 'baseline':None if root is None else root.count, 'body_reads':len(observed)-before})
                phase.reset(token)
        return invoke
    monkeypatch.setattr(ContentArchive, '_read_content', read)
    monkeypatch.setattr(TranscriptRepository, 'prepare_model_projection', wrap(TranscriptRepository.prepare_model_projection, 'projection'))
    async def lookup(ctx: AgentContext[None]) -> str:
        return 'value'
    async def model(messages, info):
        count = sum(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)
        return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id=f'call-{count}')]) if count < rounds else ModelResponse(parts=[TextPart('done')])
    features = RuntimeFeatures('review-boundaries', defaults={"execution": {"retain_history": True}})
    features.capability(TraceCapability(commit=CommitPolicy.sync()))
    features.capability(native_checkpoint())
    features.tool(lookup, effect_policy='none')
    features.agent('default', model='default', allow_tools=('lookup',))
    async with Persistence.open('review-boundaries', storage=PersistenceStorage.sqlite(tmp_path / 'boundaries.db')) as persistence_151, Runtime.open(persistence=persistence_151, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
        result = await runtime.agents.get().run('question', timeout_seconds=10)
        assert result.result.status is ExecutionStatus.SUCCEEDED
    summary = {label: {'calls': sum(item[0] == label for item in observed),
                       'object_calls': sum(item[0] == label and item[3] == 'object' for item in observed),
                       'bytes': sum(item[2] for item in observed if item[0] == label),
                       'unique': len({item[1] for item in observed if item[0] == label})}
               for label in ('projection', 'other')}
    Path(f'/tmp/review-boundary-counts-{rounds}.json').write_text(json.dumps({'summary':summary,'projection_calls':details}, sort_keys=True))
    assert summary['projection']['calls'] <= 6 * rounds + 5
    assert summary['other']['calls'] <= 3 * rounds + 2
    # Whole-kind owners are a separate cost from model-history hydration.
    # Both measured workloads (2 and 4 rounds) have 17 distinct scope/owner
    # decodes, 4 immutable identities and 5503 decoded bytes. More tool rounds
    # must not add repeated configuration/input/template decodes.
    from collections import Counter
    assert len(owner_reads) <= 17
    assert sum(item[2] for item in owner_reads) <= 5503
    assert len({item[1] for item in owner_reads}) <= 4
    assert all(transaction is not None for transaction, _, _ in owner_reads)
    assert max(Counter((transaction, identity) for transaction, identity, _ in owner_reads).values(), default=0) <= 1


@pytest.mark.asyncio
async def test_native_admission_does_not_stage_a_second_large_history_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    from linktools.ai.storage._object_sql import SqlObjectStore
    from linktools.ai.persistence.coordination import PendingTurnCoordination

    class HostAuthorization(TenantAuthorizationPolicy):
        """Exercise the real native Session and immutable input policy."""

    entered, finish = asyncio.Event(), asyncio.Event()
    large = "review-private-history-" * 10000
    calls = 0
    blocked = False
    original_put = SqlObjectStore.put

    async def put(self, *args, **kwargs):
        assert not blocked, "Native admission wrote a temporary history object"
        return await original_put(self, *args, **kwargs)

    async def forbidden_pending(*args, **kwargs):
        pytest.fail("Native Session must not publish accepted history in legacy coordination")

    monkeypatch.setattr(SqlObjectStore, "put", put)
    monkeypatch.setattr(PendingTurnCoordination, "publish", forbidden_pending)

    async def model(messages, _info):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert large in str(messages)
            entered.set()
            await finish.wait()
        return ModelResponse(parts=[TextPart(large if calls == 1 else "done")])

    namespace = "review-pending-large"
    features = RuntimeFeatures(namespace, defaults={"execution": {"retain_history": True}}, execution_commit=CommitPolicy.terminal())
    features.agent("default", model="default", allow_tools=(), allow_skills=(), allow_subagents=())
    storage = PersistenceStorage.sqlite(
        tmp_path / "pending.db", coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024),
    )
    async with (
        Persistence.open(namespace, storage=storage) as state,
        Runtime.open(persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), features),
                     authorization=HostAuthorization()) as runtime,
    ):
        agent = runtime.agents.get()
        await agent.create_session("session")
        first = await agent.session("session").start("first", idempotency_key="first")
        assert (await first.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        turns = state.session_turns
        first_row = await turns.get_turn(first.key)
        blocked = True
        try:
            execution = await agent.session("session").start("second", idempotency_key="second")
            await asyncio.wait_for(entered.wait(), 5)
            pending = await turns.get_turn(execution.key)
            assert (pending["context_from"], pending["context_through"]) == (1, 1)
            assert pending["request_json"]["prompt"] == "second"
            assert b"review-private-history-" not in canonical_json_bytes(pending["request_json"])
            assert pending["result_json"] is None
            assert await state.execution.executions.get(execution.key.turn_id, tenant_id=runtime.tenant_id) is None
        finally:
            blocked = False
            finish.set()
        assert (await execution.wait_final(timeout_seconds=10)).status is ExecutionStatus.SUCCEEDED
        assert await turns.get_turn(first.key) == first_row
        assert await state.execution.executions.get(execution.key.turn_id, tenant_id=runtime.tenant_id) is None
        history = await runtime.history.turn_session("session", principal=runtime.default_principal)
        assert [item["output"] for item in history["timeline"]["items"]] == [{"text": large}, {"text": "done"}]
