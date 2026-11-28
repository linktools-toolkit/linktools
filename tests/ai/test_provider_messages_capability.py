"""A selected provider projection is not a transcript or tool-policy mutation."""

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart

from linktools.ai.capability import AgentContext, ProviderMessagesCapability, SessionCapability, TraceCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistencePlan, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _wait_for_committed, model_features


_REF = CapabilityRef("capability", "provider-redaction", 2)


class _HostAuthorization(TenantAuthorizationPolicy):
    """Authorize the SQLite fixture’s native turns."""


def _prompts(messages):
    return [part.content for message in messages if isinstance(message, ModelRequest)
            for part in message.parts if isinstance(part, UserPromptPart)]


@pytest.mark.asyncio
async def test_selected_projection_precedes_trace_without_rewriting_turn_or_tools(tmp_path):
    input_by_turn = {}
    projected, offered, provider, origins = [], [], [], []

    def project(context, messages, tool_names):
        projected.append(context.selected_capabilities)
        offered.append(tool_names)
        for message in messages:
            if isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, UserPromptPart) and part.content == "private-selected":
                        part.content = "redacted"
        return messages

    async def inspect(context: AgentContext[None]) -> str:
        origins.append(context.original_input)
        return "tool-result"

    async def model(messages, info):
        provider.append((_prompts(messages), tuple(tool.name for tool in info.function_tools)))
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("inspect", {}, tool_call_id="inspect-1")])

    features = RuntimeFeatures("provider-messages")
    features.capability(SessionCapability(checkpoints=True))
    features.capability(TraceCapability())
    features.capability(ProviderMessagesCapability(project), id=_REF.id, revision=_REF.revision, scope="agent")
    features.tool(inspect, effect_policy="none")
    for name in ("selected", "unselected"):
        features.agent(name, model="default", allow_tools=("inspect",),
                       allow_capabilities=(_REF.id,) if name == "selected" else (),
                       allow_skills=(), allow_subagents=())
    model_features(_Models(_UsageFunctionModel(model)), features)

    storage = PersistenceStorage.sqlite(tmp_path / "provider.db")
    async with Runtime.open("provider-messages", storage=storage, features=features,
                            authorization=_HostAuthorization()) as runtime:
        actor = runtime.default_principal
        for name in ("selected", "unselected"):
            original = f"private-{name}"
            await runtime.sessions.create_turn_session(name, principal=actor)
            run = await runtime.sessions.start(name, original, agent=name, principal=actor, idempotency_key="first")
            result = await run.wait_final(timeout_seconds=10)
            assert result.status is ExecutionStatus.SUCCEEDED, result.view
            input_by_turn[name] = run.key
            history = await runtime.history.turn_result(name, run.key.turn_id, principal=actor)
            assert history["user_input"]["prompt"] == original
            interactions = await _wait_for_committed(
                lambda: runtime.history.turn_model_interactions(
                    name, run.key.turn_id, principal=actor, local_workspace=True, include_content=True,
                ),
                lambda page: len(page.items) == 2,
            )
            for item in interactions.items:
                expected = "redacted" if name == "selected" else original
                assert expected in repr(item["request"])
                if name == "selected":
                    assert original not in repr(item["request"])
        replay = await runtime.sessions.start("selected", "private-selected", agent="selected",
                                              principal=actor, idempotency_key="first")
        assert replay.key == input_by_turn["selected"]
        assert (await replay.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED

    assert projected == [(_REF,), (_REF,)]
    assert offered == [("inspect",), ("inspect",)]
    assert provider == [(["redacted"], ("inspect",))] * 2 + [(["private-unselected"], ("inspect",))] * 2
    assert [origin["prompt"]["text"] for origin in origins] == ["private-selected", "private-unselected"]


def test_runtime_wide_provider_projection_is_not_allowed():
    features = RuntimeFeatures("provider-messages")
    with pytest.raises(AIError) as error:
        features.capability(ProviderMessagesCapability(lambda _ctx, messages, _names: messages), id=_REF.id)
    assert error.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
async def test_selected_projection_requires_same_store_step_for_native_turns(tmp_path, split):
    async def model(*_args):
        pytest.fail("unsupported native topology dispatched a model")

    def project(*_args):
        pytest.fail("unsupported native topology invoked a projector")

    namespace = "provider-messages"
    features = RuntimeFeatures(namespace)
    features.capability(ProviderMessagesCapability(project), id=_REF.id, revision=_REF.revision, scope="agent")
    features.agent("default", allow_tools=(), allow_capabilities=(_REF.id,), allow_skills=(), allow_subagents=())
    model_features(_Models(_UsageFunctionModel(model)), features)
    if split:
        features.capability(SessionCapability(checkpoints=True))
        execution = StorageRoute.sqlite(tmp_path / "execution.db")
        conversation = StorageRoute.sqlite(tmp_path / "conversation.db")
        storage = PersistenceStorage(PersistencePlan(execution=execution, conversation=conversation, checkpoint=execution))
    else:
        storage = PersistenceStorage.sqlite(tmp_path / "step-off.db")

    async with Runtime.open(namespace, storage=storage, features=features,
                            authorization=_HostAuthorization()) as runtime:
        actor = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=actor)
        with pytest.raises(AIError) as error:
            await runtime.sessions.start("session", "inspect", agent="default",
                                         principal=actor, idempotency_key="blocked")
        assert error.value.code is (ErrorCode.STORAGE_CAPABILITY_MISSING if split else ErrorCode.REQUEST_FIELD_INVALID)
        if not split:
            assert error.value.safe_details["features"] == ["agent_capabilities"]
        assert (await runtime.sessions.inspect("session", principal=actor))["active_turn_id"] is None
