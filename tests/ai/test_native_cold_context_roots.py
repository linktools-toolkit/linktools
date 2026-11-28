"""Cold Session context must not hydrate bodies excluded by an admitted reset."""

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._session_transcript import decode_session_transcript_root
from linktools.ai.runtime import Runtime, RuntimeFeatures

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features


class _HostAuthorization(TenantAuthorizationPolicy):
    """Host owner for the SQLite Session fixture."""


@pytest.mark.asyncio
async def test_cold_session_hydrates_only_roots_after_successful_reset(tmp_path):
    admitted_prompts = []

    async def respond(messages, _info):
        prompts = [part.content for message in messages if isinstance(message, ModelRequest)
                   for part in message.parts if isinstance(part, UserPromptPart)]
        admitted_prompts.append(prompts)
        text = "response to " + prompts[-1]
        if prompts[-1] == "excluded-before-reset":
            text += " archived finding details" * 8192
        return ModelResponse(parts=[TextPart(text)])

    features = RuntimeFeatures("cold-context-roots")
    features.agent("default", model="default", allow_tools=(), allow_subagents=(), allow_skills=())
    model_features(_Models(_UsageFunctionModel(respond)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "cold-context.db")

    async def send(runtime, prompt, *, selection="all"):
        turn = await runtime.agents.get("default").session("thread").start(
            prompt, idempotency_key=prompt, session_context=selection,
        )
        result = await turn.wait_final(timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED, result.view
        return turn.key

    async with Persistence.open("cold-context-roots", storage=storage) as state:
        async with Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime:
            await runtime.sessions.create_turn_session("thread", principal=runtime.default_principal)
            prefix = await send(runtime, "excluded-before-reset")
            reset = await send(runtime, "reset", selection="current_turn")
            kept = await send(runtime, "retained")
            old_roots = {}
            for key in (prefix, reset, kept):
                row = await state.session_turns.get_turn(key)
                old_roots[key] = decode_session_transcript_root(row["result_json"])

    async with Persistence.open("cold-context-roots", storage=storage) as state:
        async with Runtime.open(persistence=state, features=features, authorization=_HostAuthorization()) as runtime:
            turns = state.session_turns
            assert not turns._recent
            owner, archive, capture = turns._transcript_source
            loaded = []
            hydrated_sizes = []

            async def observe(roots, *, transaction):
                loaded.append(tuple(roots))
                resolve = await capture(roots, transaction=transaction)

                async def measured():
                    values = await resolve()
                    hydrated_sizes.append(sum(len(body) for sequence in values for body in sequence))
                    return values

                return measured

            turns.bind_transcript_reader(owner, archive, observe)
            # Exact explicit range/recovery hydration still traverses all three
            # roots; only a new admission can cut history at the reset.
            await turns.read_context_range("thread", after=0, through=3)
            assert loaded.pop() == (old_roots[prefix], old_roots[reset], old_roots[kept])
            before = hydrated_sizes.pop()
            await send(runtime, "after-reopen")
            assert admitted_prompts[-1] == ["reset", "retained", "after-reopen"]
            assert loaded[0] == (old_roots[reset], old_roots[kept])
            assert before - hydrated_sizes[0] >= 192 * 1024


@pytest.mark.asyncio
async def test_cold_wide_prompts_project_then_restore_only_retained_exact_text(tmp_path):
    from sqlalchemy import event
    from .test_session_turn_sql import _BINDING
    from linktools.ai.persistence.turns import SessionTurnKey

    storage = PersistenceStorage.sqlite(tmp_path / "cold-wide.db")
    early = "旧" * 4500
    wide = "新" * 5000 + "\0" + "仍然保留"
    plain = "small\0prompt"
    leading_nul = "\0" + "零" * 6000
    async with Persistence.open("cold-wide", storage=storage) as state:
        turns = state.session_turns
        for session in ("wide", "short", "reset", "leading"):
            await turns.create(session, config={})
            prompts = {"wide": (early, early, wide, plain), "short": (plain, "short"),
                       "reset": (wide, "small reset"), "leading": (leading_nul, plain)}[session]
            for no, prompt in enumerate(prompts, 1):
                key = SessionTurnKey(session, f"{session}-{no}")
                reset = session == "reset" and no == 2
                request = {"prompt": prompt}
                if reset:
                    request["session_context"] = "current_turn"
                assert await turns.accept(
                    key, expected_revision=(no - 1) * 2, context_from=no if reset else 1, context_through=no - 1,
                    request_key=key.turn_id, request=request, binding=_BINDING, lease_seconds=60,
                )
                assert await turns.finish(
                    key, generation=1, turn_no=no, state="succeeded",
                    transcript=(ModelRequest(parts=[UserPromptPart("")]), ModelResponse(parts=[TextPart("ok")])),
                )
    async with Persistence.open("cold-wide", storage=storage) as state:
        turns = state.session_turns
        exact = await turns.read_context_range("wide", after=0, through=4)
        assert exact[-2].request["prompt"] == wide
        assert exact[-1].request["prompt"] == plain
        refills = []

        def listen(_conn, _cursor, sql, parameters, *_rest):
            if "ai_session_turns.session_key IN" in sql or "(ai_session_turns.session_key, ai_session_turns.turn_id) IN" in sql:
                refills.append(parameters)

        engine = turns._context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", listen)
        try:
            cold = await turns.recent_context("wide", through=4, max_context_bytes=8000)
            assert cold == exact[-3:]  # newest + wide predecessor + non-fitting older witness
            assert len(refills) == 1
            assert b"wide-2" in refills[0] and b"wide-3" in refills[0] and b"wide-1" not in refills[0]
            await turns.recent_context("wide", through=4, max_context_bytes=8000)
            await turns.recent_context("short", through=2, max_context_bytes=8000)
            after_reset = await turns.recent_context("reset", through=2, max_context_bytes=8000)
            assert [turn.request["prompt"] for turn in after_reset] == ["small reset"]
            assert len(refills) == 1
            leading = await turns.recent_context("leading", through=2, max_context_bytes=8000)
            assert leading[0].request["prompt"] == leading_nul
            assert len(refills) == 2 and b"leading-1" in refills[-1]
        finally:
            event.remove(engine, "before_cursor_execute", listen)
