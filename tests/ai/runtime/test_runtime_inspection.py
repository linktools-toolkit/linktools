import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec, ToolRef
from linktools.ai.capability.models import CapabilityBundle
from linktools.ai.events.payloads import SecurityDegraded
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.model.router import ModelResolver
from linktools.ai.runtime import RuntimeDependencies
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FilesystemStorage
from linktools.ai.storage.filesystem.commit import FilesystemRunCommitCoordinator

_DEGRADE_KIND = "degrade"


def _spec() -> AgentSpec:
    # A tool ref whose kind resolves to the test's fake provider, so inspect()
    # actually drives the provider through the real assembler (no private-field
    # injection of a fake assembler).
    return AgentSpec(
        id="a",
        name="a",
        model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
        tools=(ToolRef(kind=_DEGRADE_KIND, name="x"),),
    )


class _DegradingProvider:
    """Provider whose resolution degrades (the same path the real MCPProvider
    takes under best-effort discovery): emits a SecurityDegraded event through
    the context emitter and resolves no tools."""

    kind = _DEGRADE_KIND
    supported_kinds = frozenset({_DEGRADE_KIND})

    def __init__(self, *, reason: str = "tool enumeration unavailable") -> None:
        self._reason = reason

    async def resolve(self, ref, context):
        emitter = context.security_event_emitter
        if emitter is not None:
            await emitter.emit_security(
                SecurityDegraded(component="mcp-discovery", reason=self._reason)
            )
        return CapabilityBundle.empty()


def _runtime(tmp_path, *capabilities) -> Runtime:
    storage = FilesystemStorage(root=tmp_path)
    return Runtime.build(
        storage=storage,
        model_router=ModelResolver(),
        providers=RuntimeDependencies(capabilities=tuple(capabilities)),
        commit_coordinator=FilesystemRunCommitCoordinator.from_storage(storage),
    )


@pytest.mark.asyncio
async def test_inspect_surfaces_security_degradation_as_warning(tmp_path):
    rt = _runtime(tmp_path, _DegradingProvider())
    inspection = await rt.inspect(_spec())
    assert inspection.tools == ()
    assert any("security degraded" in w for w in inspection.warnings)


@pytest.mark.asyncio
async def test_inspect_warning_does_not_leak_secret_from_event(tmp_path):
    rt = _runtime(
        tmp_path,
        _DegradingProvider(
            reason="enumeration failed for https://srv/cb?token=secret-value"
        ),
    )
    inspection = await rt.inspect(_spec())
    rendered = "\n".join(inspection.warnings)
    assert "secret-value" not in rendered
    assert "security degraded" in rendered


@pytest.mark.asyncio
async def test_inspect_without_degradation_has_no_security_warnings(tmp_path):
    class _CleanProvider:
        kind = _DEGRADE_KIND
        supported_kinds = frozenset({_DEGRADE_KIND})

        async def resolve(self, ref, context):
            return CapabilityBundle.empty()

    rt = _runtime(tmp_path, _CleanProvider())
    inspection = await rt.inspect(_spec())
    assert inspection.tools == ()
    assert not any("security degraded" in w for w in inspection.warnings)


@pytest.mark.asyncio
async def test_inspection_warnings_reflect_only_security_degraded(tmp_path):
    # An observability event must not become an inspection warning -- only
    # SecurityDegraded does. Guards against the collecting emitter leaking every
    # event type into the public warnings API.
    from linktools.ai.events.payloads import ToolStarted

    class _MixedProvider:
        kind = _DEGRADE_KIND
        supported_kinds = frozenset({_DEGRADE_KIND})

        async def resolve(self, ref, context):
            em = context.security_event_emitter
            if em is not None:
                await em.emit_observability(
                    ToolStarted(tool_name="t", tool_call_id="c")
                )
                await em.emit_security(
                    SecurityDegraded(component="mcp-discovery", reason="down")
                )
            return CapabilityBundle.empty()

    rt = _runtime(tmp_path, _MixedProvider())
    inspection = await rt.inspect(_spec())
    degraded = [w for w in inspection.warnings if "security degraded" in w]
    assert len(degraded) == 1


def test_runtime_does_not_expose_executable_capability_resolver(tmp_path):
    # The CapabilityResolver carries raw executable handlers; downstream code
    # must reach tools only through inspect(), never by grabbing an assembler off
    # the runtime. There is no public capability_resolver attribute to misuse.
    rt = _runtime(tmp_path)
    assert not hasattr(rt, "capability_resolver")
    assert callable(getattr(rt, "inspect", None))


@pytest.mark.asyncio
async def test_inspection_does_not_leak_handlers_or_managed_definitions(tmp_path):
    from linktools.ai.tool.models import (
        ManagedToolDefinition,
        ToolContribution,
        ToolDescriptor,
    )

    handler_calls: "list" = []

    async def _handler(**arguments):
        handler_calls.append(arguments)
        return "ok"

    descriptor = ToolDescriptor(
        name="t", source="test", category="c", risk="low", mutating=False
    )
    definition = ManagedToolDefinition(
        descriptor=descriptor, handler=_handler, parameters_json_schema={}
    )
    bundle = CapabilityBundle(
        tool_contributions=(ToolContribution(tools=(definition,)),), prompt_sections={}
    )

    class _BundleProvider:
        kind = _DEGRADE_KIND
        supported_kinds = frozenset({_DEGRADE_KIND})

        async def resolve(self, ref, context):
            return bundle

    rt = _runtime(tmp_path, _BundleProvider())
    inspection = await rt.inspect(_spec())

    # Only safe ToolDescriptors are exposed; the handler and the
    # ManagedToolDefinition that carries it are not reachable from the snapshot.
    assert inspection.tools == (descriptor,)
    assert all(not hasattr(t, "handler") for t in inspection.tools)
    rendered = repr(inspection)
    assert "_handler" not in rendered
    assert "ManagedToolDefinition" not in rendered
    # The handler was never invoked by inspection.
    assert handler_calls == []
