import pytest

from linktools.ai.agent.assembly import (
    AgentAssembler,
    AgentContribution,
    AgentFeatureContext,
    AgentFeatureRef,
    AgentFeatureRegistry,
)
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.agent.tool.schema import JsonSchemaToolValidator
from linktools.ai.agent.tool.exposure import (
    ToolAssembler,
    ToolExposurePolicy,
)
from linktools.ai.agent.tool.models import (
    ToolCategory,
    ToolDefinition,
    ToolDescriptor,
    ToolSource,
)
from linktools.ai.errors import (
    AgentAssemblyError,
    AgentFeatureConflictError,
    AgentFeatureNotFoundError,
    ToolConflictError,
)
from linktools.ai.governance.policy.rule import RiskLevel, SideEffectKind
from linktools.ai.model.policy import ModelPolicy


def _context() -> AgentFeatureContext:
    return AgentFeatureContext(
        agent_id="agent",
        execution_id="execution",
        root_execution_id="execution",
        parent_execution_id=None,
        session_id="session",
        tenant_id="tenant",
        user_id="user",
        workspace=None,
        sandbox=None,
    )


def _spec(*features: AgentFeatureRef) -> AgentSpec:
    return AgentSpec(
        id="agent",
        name="agent",
        model=ModelPolicy(primary="test"),
        instructions=PromptSpec("test"),
        features=features,
    )


async def _handler() -> str:
    return "ok"


def _tool(
    ref: AgentFeatureRef,
    name: str,
    *,
    side_effect: SideEffectKind = SideEffectKind.READ_ONLY,
) -> ToolDefinition:
    return ToolDefinition(
        descriptor=ToolDescriptor(
            name=name,
            source=ToolSource.EXTENSION,
            category=ToolCategory.EXTENSION_READ,
            risk=RiskLevel.LOW,
            side_effect=side_effect,
            feature=ref,
        ),
        handler=_handler,
        input_schema={"type": "object", "properties": {}},
    )


class Provider:
    supported_kinds = ("test",)

    def __init__(self, contribution: AgentContribution) -> None:
        self.contribution = contribution

    async def resolve(self, ref, context):
        return self.contribution


def _assembler(
    providers: tuple[object, ...] = (),
    *,
    exposure: ToolExposurePolicy = ToolExposurePolicy(),
) -> tuple[AgentAssembler, AgentFeatureRegistry]:
    registry = AgentFeatureRegistry()
    for provider in providers:
        registry.register(provider)
    registry.freeze()
    return (
        AgentAssembler(
            registry=registry,
            tool_assembler=ToolAssembler(
                exposure=exposure,
                schema_validator=JsonSchemaToolValidator(),
            ),
        ),
        registry,
    )


@pytest.mark.asyncio
async def test_empty_features_produce_empty_assembly() -> None:
    assembler, _ = _assembler()
    assembly = await assembler.assemble(_spec(), _context())
    assert assembly.tools == ()
    assert dict(assembly.prompt_sections) == {}


@pytest.mark.asyncio
async def test_assembly_emits_feature_resolve_events() -> None:
    class Sink:
        def __init__(self):
            self.events = []

        async def emit(self, event):
            self.events.append(event)

    ref = AgentFeatureRef("test", "one")
    sink = Sink()
    registry = AgentFeatureRegistry()
    registry.register(Provider(AgentContribution.empty()))
    registry.freeze()
    assembler = AgentAssembler(
        registry=registry,
        tool_assembler=ToolAssembler(
            exposure=ToolExposurePolicy(),
            schema_validator=JsonSchemaToolValidator(),
        ),
        events=sink,
    )
    await assembler.assemble(_spec(ref), _context())
    assert [type(event).__name__ for event in sink.events] == [
        "AgentFeatureResolveStarted",
        "AgentFeatureResolveCompleted",
    ]


@pytest.mark.asyncio
async def test_prompt_sections_merge_in_declaration_order() -> None:
    first = AgentFeatureRef("test", "first")
    second = AgentFeatureRef("test", "second")

    class OrderedProvider:
        supported_kinds = ("test",)

        async def resolve(self, ref, context):
            return AgentContribution(prompt_sections={"shared": ref.name})

    assembler, _ = _assembler((OrderedProvider(),))
    assembly = await assembler.assemble(
        _spec(first, second),
        _context(),
    )
    assert assembly.prompt_sections["shared"] == "first\nsecond"


@pytest.mark.asyncio
async def test_duplicate_feature_is_rejected() -> None:
    ref = AgentFeatureRef("test", "same")
    assembler, _ = _assembler((Provider(AgentContribution.empty()),))
    with pytest.raises(AgentFeatureConflictError):
        await assembler.assemble(_spec(ref, ref), _context())


@pytest.mark.asyncio
async def test_missing_provider_is_rejected() -> None:
    assembler, _ = _assembler()
    with pytest.raises(AgentFeatureNotFoundError):
        await assembler.assemble(
            _spec(AgentFeatureRef("missing", "one")),
            _context(),
        )


@pytest.mark.asyncio
async def test_tool_conflict_is_rejected() -> None:
    first = AgentFeatureRef("test", "first")
    second = AgentFeatureRef("test", "second")

    class CollidingProvider:
        supported_kinds = ("test",)

        async def resolve(self, ref, context):
            return AgentContribution(tools=(_tool(ref, "same"),))

    assembler, _ = _assembler((CollidingProvider(),))
    with pytest.raises(ToolConflictError):
        await assembler.assemble(_spec(first, second), _context())


@pytest.mark.asyncio
async def test_mutating_tool_hidden_by_default_and_exposed_explicitly() -> None:
    ref = AgentFeatureRef("test", "one")
    contribution = AgentContribution(
        tools=(
            _tool(
                ref,
                "write",
                side_effect=SideEffectKind.NAMESPACE_MUTATING,
            ),
        )
    )
    hidden, _ = _assembler((Provider(contribution),))
    assert (await hidden.assemble(_spec(ref), _context())).tools == ()

    exposed, _ = _assembler(
        (Provider(contribution),),
        exposure=ToolExposurePolicy(expose_execution_tools=True),
    )
    assert len((await exposed.assemble(_spec(ref), _context())).tools) == 1


def test_registry_freeze_rejects_mutation() -> None:
    _, registry = _assembler((Provider(AgentContribution.empty()),))
    with pytest.raises(AgentAssemblyError):
        registry.replace(Provider(AgentContribution.empty()))


@pytest.mark.asyncio
async def test_tool_count_limit_is_enforced() -> None:
    ref = AgentFeatureRef("test", "one")
    assembler, _ = _assembler(
        (
            Provider(
                AgentContribution(
                    tools=(_tool(ref, "one"), _tool(ref, "two"))
                )
            ),
        ),
        exposure=ToolExposurePolicy(max_tools_total=1),
    )
    with pytest.raises(ToolConflictError):
        await assembler.assemble(_spec(ref), _context())
