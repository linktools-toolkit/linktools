import pytest

from linktools.ai.execution.commands import ClaimExecution, CompleteExecution, StartExecution
from linktools.ai.execution.domain import ApprovalDecision, MessageCaptureState, RunApproval, RunDefinition, RunKind, RunnableType, RunStatus, RunUsage, compute_run_definition_hash
from linktools.ai.execution.persistence.local import LocalExecutionBackend
from linktools.ai.execution.snapshots import AgentSnapshotData
from linktools.ai.execution.query import ExecutionQueryService
from linktools.ai.governance.identity import ActorRef, PrincipalContext, ScopeSet


def principal(user: str) -> PrincipalContext:
    return PrincipalContext("tenant", user, ActorRef("user", user), ScopeSet.allow_all())


def principal_without_inspect(user: str) -> PrincipalContext:
    return PrincipalContext(
        "tenant",
        user,
        ActorRef("user", user),
        ScopeSet.of("execution:run"),
    )


def _definition() -> RunDefinition:
    schema = "agent-spec.v1"
    spec = {"id": "agent"}
    return RunDefinition(
        "agent",
        RunnableType.AGENT,
        schema,
        spec,
        compute_run_definition_hash(schema=schema, spec=spec),
    )


@pytest.mark.asyncio
async def test_query_service_authorizes_before_returning_turns(tmp_path):
    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "secret"))
    query = ExecutionQueryService(store)
    with pytest.raises(Exception):
        await query.list_session_turns(session_id="s", principal=principal("other"))
    page = await query.list_session_turns(session_id="s", principal=principal("u"))
    assert page.items[0].input == "secret"


@pytest.mark.asyncio
async def test_run_detail_effective_input_comes_from_run_record_input(tmp_path):
    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "secret"))
    query = ExecutionQueryService(store)
    detail = await query.get_run_detail(run_id="r", principal=principal("u"))
    assert detail.effective_input == "secret"


@pytest.mark.asyncio
async def test_query_requires_inspect_scope(tmp_path):
    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "secret"))
    query = ExecutionQueryService(store)
    with pytest.raises(Exception):
        await query.get_run_detail(
            run_id="r",
            principal=principal_without_inspect("u"),
        )


async def _complete_run_with_messages(store, run_id, owner, fence, messages):
    # Persist a terminal snapshot carrying the given delta_messages, so a run
    # ends up with a retrievable turn delta. Mirrors what ExecutionService does.
    await store.complete_run(
        CompleteExecution(
            run_id,
            owner,
            fence,
            AgentSnapshotData(messages, None, RunUsage(), len(messages)),
            0,
        )
    )


@pytest.mark.asyncio
async def test_get_run_messages_returns_turn_delta(tmp_path):
    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "hi"))
    from datetime import datetime, timezone, timedelta

    claimed = await store.claim_run(ClaimExecution("r", "runtime", datetime.now(timezone.utc), timedelta(minutes=5)))
    messages = ({"kind": "request", "parts": [{"type": "user_prompt", "content": "hi"}]},)
    await _complete_run_with_messages(store, "r", claimed.lease.owner, claimed.lease.fence, messages)
    query = ExecutionQueryService(store)
    result = await query.get_run_messages(run_id="r", principal=principal("u"))
    assert result == messages


@pytest.mark.asyncio
async def test_get_run_messages_empty_for_run_without_snapshot(tmp_path):
    # A run still PENDING (never completed) has no persisted snapshot -> ().
    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "hi"))
    query = ExecutionQueryService(store)
    assert await query.get_run_messages(run_id="r", principal=principal("u")) == ()


@pytest.mark.asyncio
async def test_get_run_messages_hides_unknown_run_existence(tmp_path):
    store = LocalExecutionBackend(tmp_path)
    query = ExecutionQueryService(store)
    with pytest.raises(Exception):
        await query.get_run_messages(run_id="nonexistent", principal=principal("u"))


@pytest.mark.asyncio
async def test_get_run_messages_requires_inspect_scope(tmp_path):
    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "hi"))
    query = ExecutionQueryService(store)
    with pytest.raises(Exception):
        await query.get_run_messages(
            run_id="r",
            principal=principal_without_inspect("u"),
        )


@pytest.mark.asyncio
async def test_get_run_messages_round_trips_through_decode(tmp_path):
    # The returned JsonValue reconstructs into real pydantic-ai ModelMessage
    # objects via decode_model_messages -- proving the losslessness claim.
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from linktools.ai.execution.trace_codec import encode_model_messages, decode_model_messages

    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "hi"))
    from datetime import datetime, timezone, timedelta

    claimed = await store.claim_run(ClaimExecution("r", "runtime", datetime.now(timezone.utc), timedelta(minutes=5)))
    original = (ModelRequest(parts=[UserPromptPart(content="hello")]),)
    encoded = encode_model_messages(original)
    await _complete_run_with_messages(store, "r", claimed.lease.owner, claimed.lease.fence, encoded)
    query = ExecutionQueryService(store)
    fetched = await query.get_run_messages(run_id="r", principal=principal("u"))
    decoded = decode_model_messages(fetched)
    assert len(decoded) == 1
    assert isinstance(decoded[0], ModelRequest)
    assert isinstance(decoded[0].parts[0], UserPromptPart)
    assert decoded[0].parts[0].content == "hello"


# ---- acceptance tests: state filtering + non-terminal recovery -------------


async def _run_to_paused(store, run_id, owner, fence, partial, approval):
    # Persist a PAUSED snapshot: delta = m_partial, checkpoint = all_messages.
    from linktools.ai.execution.commands import PauseExecution
    await store.pause_run(
        PauseExecution(run_id, owner, fence, AgentSnapshotData(partial, None, RunUsage(), len(partial), MessageCaptureState.COMPLETE, partial), approval, 0)
    )


@pytest.mark.asyncio
async def test_paused_then_cancelled_keeps_delta_clears_checkpoint_excludes_from_context(tmp_path):
    # PAUSED->CANCELLED: TURN_DELTA retained (audit), RESUME_CHECKPOINT
    # cleared, load_session_context excludes the turn, get_run_messages returns
    # the paused delta.
    from datetime import datetime, timezone, timedelta
    from linktools.ai.execution.commands import RequestCancellation
    from linktools.ai.execution.domain import RunApproval

    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "hi"))
    claimed = await store.claim_run(ClaimExecution("r", "runtime", datetime.now(timezone.utc), timedelta(minutes=5)))
    m_partial = ({"kind": "request", "parts": [{"type": "user_prompt", "content": "hi"}]},)
    approval = RunApproval("ap1", "tc1", "tool", "fp")
    await _run_to_paused(store, "r", claimed.lease.owner, claimed.lease.fence, m_partial, approval)
    # Direct cancel from PAUSED.
    await store.request_cancel(RequestCancellation("r", claimed.lease.owner or "runtime", claimed.lease.fence, datetime.now(timezone.utc)))
    # checkpoint cleared.
    snapshot = await store.get_snapshot("r")
    assert snapshot is not None
    assert snapshot.resume_messages == ()
    # delta retained + audit-recoverable.
    query = ExecutionQueryService(store)
    assert await query.get_run_messages(run_id="r", principal=principal("u")) == m_partial
    # excluded from Session Context (CANCELLED not COMPLETED).
    assert await store.load_session_context("s") == ()
    # capture_state COMPLETE (paused point was a stable commit).
    turns = await store.get_session_messages("s")
    assert turns[0].capture_state is MessageCaptureState.COMPLETE
    assert turns[0].status is RunStatus.CANCELLED


@pytest.mark.asyncio
async def test_paused_then_resumed_completed_accumulates_delta_clears_checkpoint(tmp_path):
    # PAUSED->RESUME->COMPLETED: m_partial + m_resume appear once, the turn
    # enters Session Context, checkpoint cleared.
    from datetime import datetime, timezone, timedelta
    from linktools.ai.execution.commands import ResumeExecution, CompleteExecution

    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "hi"))
    claimed = await store.claim_run(ClaimExecution("r", "runtime", datetime.now(timezone.utc), timedelta(minutes=5)))
    m_partial = ({"kind": "request", "parts": [{"type": "user_prompt", "content": "hi"}]},)
    approval = RunApproval("ap1", "tc1", "tool", "fp", ApprovalDecision.ALLOW, "u", datetime.now(timezone.utc))
    await _run_to_paused(store, "r", claimed.lease.owner, claimed.lease.fence, m_partial, approval)
    # resume (PAUSED->PENDING) then complete with m_resume.
    await store.resume_run(ResumeExecution("r"))
    reclaimed = await store.claim_run(ClaimExecution("r", "runtime", datetime.now(timezone.utc), timedelta(minutes=5)))
    m_resume = ({"kind": "response", "parts": [{"type": "text", "content": "done"}]},)
    await store.complete_run(
        CompleteExecution("r", reclaimed.lease.owner, reclaimed.lease.fence, AgentSnapshotData(m_resume, None, RunUsage(), len(m_partial) + len(m_resume)), 1)
    )
    query = ExecutionQueryService(store)
    # delta = m_partial + m_resume (appended once).
    assert await query.get_run_messages(run_id="r", principal=principal("u")) == m_partial + m_resume
    # Session Context now includes this COMPLETED turn.
    assert await store.load_session_context("s") == m_partial + m_resume
    # checkpoint cleared.
    snapshot = await store.get_snapshot("r")
    assert snapshot is not None
    assert snapshot.resume_messages == ()


@pytest.mark.asyncio
async def test_get_session_messages_returns_all_turns_grouped_with_capture_state(tmp_path):
    # Audit History: every turn (any status) grouped per turn with status +
    # capture_state. Two COMPLETED turns + one CANCELLED all appear.
    from datetime import datetime, timezone, timedelta
    from linktools.ai.execution.commands import RequestCancellation

    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    # turn 1: COMPLETED.
    await store.start_run(StartExecution("r1", "s", RunKind.USER_TURN, definition, "q1"))
    c1 = await store.claim_run(ClaimExecution("r1", "w", datetime.now(timezone.utc), timedelta(minutes=5)))
    await store.complete_run(CompleteExecution("r1", "w", c1.lease.fence, AgentSnapshotData(({"m": 1},), None, RunUsage(), 1), 0))
    # turn 2: CANCELLED (direct from PENDING).
    await store.start_run(StartExecution("r2", "s", RunKind.USER_TURN, definition, "q2"))
    await store.request_cancel(RequestCancellation("r2", "runtime", 0, datetime.now(timezone.utc)))
    query = ExecutionQueryService(store)
    views = await query.get_session_messages(session_id="s", principal=principal("u"))
    assert len(views) == 2
    assert views[0].status is RunStatus.COMPLETED
    assert views[0].capture_state is MessageCaptureState.COMPLETE
    assert views[0].messages == ({"m": 1},)
    assert views[1].status is RunStatus.CANCELLED
    assert views[1].capture_state is MessageCaptureState.COMPLETE
    assert views[1].messages == ()


@pytest.mark.asyncio
async def test_multi_turn_session_context_reconstructs_full_history_no_redundancy(tmp_path):
    # Core correctness: across N completed turns, (1) load_session_context
    # returns the concatenation of every turn's delta in order (full history
    # reconstruction); (2) each message appears in exactly one turn's delta (no
    # O(N^2) cumulative duplication); (3) get_run_messages(turn_N) returns only
    # that turn's delta, not prior turns'.
    from datetime import datetime, timezone, timedelta

    store = LocalExecutionBackend(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = _definition()
    per_turn_delta = []
    run_ids = []
    for n in range(1, 4):
        run_id = f"r{n}"
        await store.start_run(StartExecution(run_id, "s", RunKind.USER_TURN, definition, f"q{n}"))
        claimed = await store.claim_run(ClaimExecution(run_id, "w", datetime.now(timezone.utc), timedelta(minutes=5)))
        delta = ({"kind": "request", "parts": [{"type": "user_prompt", "content": f"q{n}"}]},
                 {"kind": "response", "parts": [{"type": "text", "content": f"a{n}"}]})
        per_turn_delta.append(delta)
        run_ids.append(run_id)
        await store.complete_run(CompleteExecution(run_id, "w", claimed.lease.fence, AgentSnapshotData(delta, f"out{n}", RunUsage(), len(delta)), 0))
    query = ExecutionQueryService(store)
    # (1) full history = turn1 + turn2 + turn3 deltas, in order.
    expected_full = per_turn_delta[0] + per_turn_delta[1] + per_turn_delta[2]
    assert await store.load_session_context("s") == expected_full
    # (2) no redundancy: each turn's delta is disjoint (no message appears twice
    # across turns -- the O(N^2) failure mode would embed turn1 in turn2).
    assert per_turn_delta[0] and per_turn_delta[1] and per_turn_delta[2]
    all_msgs = expected_full
    assert len(all_msgs) == len(set(map(_msg_key, all_msgs)))  # no dupes
    # (3) per-turn isolation: get_run_messages(turn2) == turn2's delta only.
    assert await query.get_run_messages(run_id="r2", principal=principal("u")) == per_turn_delta[1]
    # snapshot carries no cumulative history (checkpoint cleared on COMPLETED).
    for rid in run_ids:
        snapshot = await store.get_snapshot(rid)
        assert snapshot is not None
        assert snapshot.resume_messages == ()


def _msg_key(m):
    # Stable identity for duplicate detection across the multi-turn history.
    parts = tuple((p.get("type"), p.get("content")) for p in m.get("parts", ()))
    return (m.get("kind"), parts)
