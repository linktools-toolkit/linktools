"""Tools see exact capability identities from their admitted Agent binding."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturn, ToolReturnPart

from linktools.ai.capability import AgentContext, ReferencedInputsCapability, SessionCapability
from linktools.ai.core import ExecutionStatus, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistencePlan, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_native_referenced_tool_delivery import _IMAGE, _SOURCE, _reference


_SELECTED = CapabilityRef("capability", "image-loader", 3)


class _HostAuthorization(TenantAuthorizationPolicy):
    """Authorize the SQL fixtures' session operations."""


def _features(seen, *, checkpoints=False):
    class BoundCapability(AbstractCapability):
        pass

    async def inspect(context: AgentContext[None]) -> str:
        seen.append(context)
        return "ready"

    async def model(messages, _info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("inspect", {}, tool_call_id="inspect-1")])

    features = RuntimeFeatures("admitted-tool-capabilities")
    if checkpoints:
        features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.capability(BoundCapability(), id=_SELECTED.id, revision=_SELECTED.revision, scope="agent")
    features.tool(inspect, effect_policy="none")
    for name, allowed in (("enabled", (_SELECTED.id,)), ("disabled", ())):
        features.agent(name, model="default", allow_tools=("inspect",), allow_capabilities=allowed,
                       allow_skills=(), allow_subagents=())
    return model_features(_Models(_UsageFunctionModel(model)), features)


@pytest.mark.asyncio
async def test_standalone_tool_context_has_selected_identity_and_revision():
    seen = []
    async with Runtime.open(
        "admitted-tool-capabilities", storage=PersistenceStorage.in_memory(), features=_features(seen),
    ) as runtime:
        for name in ("enabled", "disabled"):
            run = await runtime.agents.get(name).start("inspect")
            assert (await run.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED

    [selected, absent] = seen
    assert selected.selected_capabilities == (_SELECTED,)
    assert CapabilityRef("capability", _SELECTED.id, 2) not in selected.selected_capabilities
    assert absent.selected_capabilities == ()
    with pytest.raises(FrozenInstanceError):
        selected.selected_capabilities = ()


@pytest.mark.asyncio
async def test_native_tool_context_omits_registered_but_unselected_capability(tmp_path):
    seen = []
    async with Runtime.open(
        "admitted-tool-capabilities", storage=PersistenceStorage.sqlite(tmp_path / "context.db"),
        features=_features(seen, checkpoints=True), authorization=_HostAuthorization(),
    ) as runtime:
        actor = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=actor)
        handle = await runtime.sessions.start_turn("session", "inspect", agent="disabled", principal=actor,
                                              idempotency_key="allowed")
        assert (await handle.wait_final(timeout_seconds=5)).status is ExecutionStatus.SUCCEEDED
        [context] = seen
        assert context.selected_capabilities == ()

        with pytest.raises(AIError) as unsupported:
            await runtime.sessions.start_turn("session", "inspect", agent="enabled", principal=actor,
                                         idempotency_key="unsupported")
        assert unsupported.value.code is ErrorCode.REQUEST_FIELD_INVALID
        assert unsupported.value.safe_details["features"] == ["agent_capabilities"]
        assert seen == [context]
        assert (await runtime.sessions.inspect("session", principal=actor))["active_turn_id"] is None



@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("unselected", "selected", "wrong_revision"))
async def test_native_step_only_loads_exact_selected_reference_and_replay_is_inert(tmp_path, mode):
    selected = mode != "unselected"
    namespace = "native-selected-reference"
    observed, loads = [], []
    reference = _reference(_IMAGE)
    reference.origin = None
    if mode == "wrong_revision":
        reference.source = CapabilityRef(_SOURCE.kind, _SOURCE.id, _SOURCE.revision + 1)

    async def lookup(context: AgentContext[None]) -> ToolReturn | str:
        observed.append((context.execution_id, context.selected_capabilities))
        if _SOURCE not in context.selected_capabilities:
            return "image unavailable"
        return ToolReturn(return_value=[reference])

    async def load(context: AgentContext[None], content) -> bytes:
        assert context.principal == actor and context.selected_capabilities == (_SOURCE,)
        assert content.origin is not None and content.origin.principal == actor
        assert (content.origin.execution_id, content.origin.tool_call_id) == (context.execution_id, "image-call")
        assert content.ref == reference.ref
        loads.append(content)
        return _IMAGE

    async def model(messages, info):
        results = [part for message in messages if isinstance(message, ModelRequest)
                   for part in message.parts if isinstance(part, ToolReturnPart)]
        if not results:
            return ModelResponse(parts=[ToolCallPart("lookup", {}, tool_call_id="image-call")])
        [result] = results
        assert mode != "wrong_revision", "unselected source revision reached the provider"
        if selected:
            [image] = result.content
            assert isinstance(image, BinaryContent) and image.data == _IMAGE
        else:
            assert result.content == "image unavailable"
        return ModelResponse(parts=[TextPart("done")])

    features = RuntimeFeatures(namespace)
    features.capability(SessionCapability(checkpoints=True), scope="runtime")
    features.capability(ReferencedInputsCapability(load), id=_SOURCE.id, revision=_SOURCE.revision, scope="agent")
    features.tool(lookup, effect_policy="none")
    features.agent("default", allow_tools=("lookup",), allow_capabilities=(_SOURCE.id,) if selected else (),
                   allow_skills=(), allow_subagents=())
    models = _Models(_UsageFunctionModel(model))
    models.contract = {**models.contract, "input_mime_types": ["image/png"]}
    model_features(models, features)
    storage = PersistenceStorage.sqlite(tmp_path / "native-selected.db")
    async with Runtime.open(namespace, storage=storage, features=features, authorization=_HostAuthorization()) as runtime:
        actor = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=actor)
        first = await runtime.sessions.start_turn("session", "Read evidence", principal=actor, agent="default",
                                             idempotency_key="first")
        final = await first.wait_final(timeout_seconds=10)
        if mode == "wrong_revision":
            assert final.status is ExecutionStatus.FAILED
            assert final.view["error_code"] == ErrorCode.CAPABILITY_RESOLUTION_INVALID
        else:
            assert final.status is ExecutionStatus.SUCCEEDED and final.view["output"] == {"text": "done"}
    async with Runtime.open(namespace, storage=storage, features=features, authorization=_HostAuthorization()) as runtime:
        replay = await runtime.sessions.start_turn("session", "Read evidence", principal=actor, agent="default",
                                              idempotency_key="first")
        assert replay.key == first.key
        assert (await replay.wait_final(timeout_seconds=5)).view == final.view
    assert observed == [(first.key.turn_id, (_SOURCE,) if selected else ())]
    assert len(loads) == int(mode == "selected")



@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
async def test_referenced_input_capability_remains_unavailable_without_same_store_step(tmp_path, split):
    namespace = "native-restricted-reference"

    async def forbidden(*args):
        pytest.fail("unsupported native topology may not reach the loader or provider")

    features = RuntimeFeatures(namespace)
    features.capability(ReferencedInputsCapability(forbidden), id=_SOURCE.id, revision=_SOURCE.revision, scope="agent")
    features.agent("default", allow_tools=(), allow_capabilities=(_SOURCE.id,), allow_skills=(), allow_subagents=())
    model_features(_Models(_UsageFunctionModel(forbidden)), features)
    if split:
        features.capability(SessionCapability(checkpoints=True), scope="runtime")
        execution = StorageRoute.sqlite(tmp_path / "execution.db")
        storage = PersistenceStorage(PersistencePlan(
            conversation=StorageRoute.sqlite(tmp_path / "conversation.db"),
            execution=execution, checkpoint=execution,
        ))
    else:
        storage = PersistenceStorage.sqlite(tmp_path / "step-off.db")
    async with Runtime.open(namespace, storage=storage, features=features, authorization=_HostAuthorization()) as runtime:
        actor = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=actor)
        before = await runtime.history.turn_session("session", principal=actor)
        with pytest.raises(AIError) as rejected:
            await runtime.sessions.start_turn("session", "Do not dispatch", agent="default", principal=actor,
                                         idempotency_key="forbidden")
        assert rejected.value.code is (ErrorCode.STORAGE_CAPABILITY_MISSING if split else ErrorCode.REQUEST_FIELD_INVALID)
        assert await runtime.history.turn_session("session", principal=actor) == before
