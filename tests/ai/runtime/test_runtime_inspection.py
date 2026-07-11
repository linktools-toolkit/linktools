import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.capability.bundle import CapabilityBundle
from linktools.ai.events.payloads import SecurityDegraded
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import Runtime
from linktools.ai.storage.facade import FileStorage


def _spec() -> AgentSpec:
    return AgentSpec(
        id="a", name="a", model=ModelPolicy(primary="m"),
        instructions=PromptSpec(instructions="hi"),
    )


class _DegradingAssembler:
    """Stand-in for an assembler whose resolution degrades (the same path the
    real MCPProvider takes under best-effort discovery): it emits a
    SecurityDegraded event through the context emitter and resolves no tools."""

    def __init__(self, *, reason: str = "tool enumeration unavailable") -> None:
        self._reason = reason

    async def assemble(self, spec, context):
        emitter = context.security_event_emitter
        if emitter is not None:
            await emitter.emit_security(
                SecurityDegraded(component="mcp-discovery", reason=self._reason))
        return CapabilityBundle.empty()


def _runtime(tmp_path) -> Runtime:
    from linktools.ai.model.router import ModelRouter
    return Runtime.build(storage=FileStorage(root=tmp_path), model_router=ModelRouter())


@pytest.mark.asyncio
async def test_inspect_surfaces_security_degradation_as_warning(tmp_path):
    rt = _runtime(tmp_path)
    rt._capability_assembler = _DegradingAssembler()
    inspection = await rt.inspect(_spec(), execution=None)
    assert inspection.tools == ()
    assert any("security degraded" in w for w in inspection.warnings)


@pytest.mark.asyncio
async def test_inspect_warning_does_not_leak_secret_from_event(tmp_path):
    rt = _runtime(tmp_path)
    rt._capability_assembler = _DegradingAssembler(
        reason="enumeration failed for https://srv/cb?token=secret-value")
    inspection = await rt.inspect(_spec(), execution=None)
    rendered = "\n".join(inspection.warnings)
    assert "secret-value" not in rendered
    assert "security degraded" in rendered


@pytest.mark.asyncio
async def test_inspect_without_degradation_has_no_security_warnings(tmp_path):
    class _CleanAssembler:
        async def assemble(self, spec, context):
            return CapabilityBundle.empty()
    rt = _runtime(tmp_path)
    rt._capability_assembler = _CleanAssembler()
    inspection = await rt.inspect(_spec(), execution=None)
    assert inspection.tools == ()
    assert not any("security degraded" in w for w in inspection.warnings)


@pytest.mark.asyncio
async def test_inspection_warnings_reflect_only_security_degraded(tmp_path):
    # An observability event must not become an inspection warning -- only
    # SecurityDegraded does. Guards against the collecting emitter leaking every
    # event type into the public warnings API.
    from linktools.ai.events.payloads import ToolStarted

    class _MixedAssembler:
        async def assemble(self, spec, context):
            em = context.security_event_emitter
            if em is not None:
                await em.emit_observability(ToolStarted(tool_name="t", tool_call_id="c"))
                await em.emit_security(
                    SecurityDegraded(component="mcp-discovery", reason="down"))
            return CapabilityBundle.empty()

    rt = _runtime(tmp_path)
    rt._capability_assembler = _MixedAssembler()
    inspection = await rt.inspect(_spec(), execution=None)
    degraded = [w for w in inspection.warnings if "security degraded" in w]
    assert len(degraded) == 1


def test_runtime_does_not_expose_executable_capability_assembler(tmp_path):
    # The CapabilityAssembler carries raw executable handlers; downstream code
    # must reach tools only through inspect(), never by grabbing an assembler off
    # the runtime. There is no public capability_assembler attribute to misuse.
    rt = _runtime(tmp_path)
    assert not hasattr(rt, "capability_assembler")
    assert callable(getattr(rt, "inspect", None))


@pytest.mark.asyncio
async def test_inspection_does_not_leak_handlers_or_managed_definitions(tmp_path):
    from linktools.ai.security.descriptor import ToolDescriptor
    from linktools.ai.tool.contribution import ManagedToolDefinition, ToolContribution

    handler_calls: "list" = []

    async def _handler(**arguments):
        handler_calls.append(arguments)
        return "ok"

    descriptor = ToolDescriptor(
        name="t", source="test", category="c", risk="low", mutating=False)
    definition = ManagedToolDefinition(
        descriptor=descriptor, handler=_handler, parameters_json_schema={})
    bundle = CapabilityBundle(
        tool_contributions=(ToolContribution(tools=(definition,)),),
        prompt_sections={})

    class _Assembler:
        async def assemble(self, spec, context):
            return bundle

    rt = _runtime(tmp_path)
    rt._capability_assembler = _Assembler()
    inspection = await rt.inspect(_spec(), execution=None)

    # Only safe ToolDescriptors are exposed; the handler and the
    # ManagedToolDefinition that carries it are not reachable from the snapshot.
    assert inspection.tools == (descriptor,)
    assert all(not hasattr(t, "handler") for t in inspection.tools)
    rendered = repr(inspection)
    assert "_handler" not in rendered
    assert "ManagedToolDefinition" not in rendered
    # The handler was never invoked by inspection.
    assert handler_calls == []
