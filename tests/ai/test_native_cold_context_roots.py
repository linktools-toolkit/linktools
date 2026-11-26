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
        turn = await runtime.agents.get("default").start(
            prompt, session_id="thread", idempotency_key=prompt, session_context=selection,
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
