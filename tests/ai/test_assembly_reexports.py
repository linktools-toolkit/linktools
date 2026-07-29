def test_assembly_public_exports() -> None:
    from linktools.ai.agent.assembly import (
        AgentAssembler,
        AgentContribution,
        AgentFeatureContext,
        AgentFeatureProvider,
        AgentFeatureRef,
        AgentFeatureRegistry,
        AgentInspection,
    )

    assert all(
        value is not None
        for value in (
            AgentAssembler,
            AgentContribution,
            AgentFeatureContext,
            AgentFeatureProvider,
            AgentFeatureRef,
            AgentFeatureRegistry,
            AgentInspection,
        )
    )
