#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_preflight must fail closed when an agent assembles tools on a runtime that
was built without a ToolStateStore / ToolPolicyResolver. The gate used to raise
``NameError`` because ``RuntimeInitializationError`` was referenced without
being imported; it must raise the typed error so a caller can catch it and
report that tools are not configured rather than crashing on an unbound name."""

import pytest

from linktools.ai.agent.assembly.models import AgentAssembly
from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.errors import RuntimeInitializationError
from linktools.ai.execution.service import ExecutionService
from linktools.ai.governance.identity import trusted_local_principal
from linktools.ai.model.policy import ModelPolicy


class _AssemblerWithTools:
    """Stand-in assembler whose assembled surface always carries one tool, so
    the not-ready gate in ``_preflight`` is the only behavior under test -- the
    real store/compiler/engine are never reached because the gate raises first."""

    def validate_features(self, spec: AgentSpec) -> None:
        return None

    async def assemble(self, spec: AgentSpec, context: object) -> AgentAssembly:
        return AgentAssembly(prompt_sections={}, tools=(object(),), feature_owners={})


@pytest.mark.asyncio
async def test_tool_bearing_agent_on_not_ready_runtime_raises_typed_error() -> None:
    service = ExecutionService(
        store=None,
        compiler=None,
        engine=None,
        assembler=_AssemblerWithTools(),
        tool_execution_ready=False,
        sandbox=None,
        spec_codec=None,
        authorization=None,
        live_events=None,
        security_events=None,
    )
    spec = AgentSpec("agent", "agent", ModelPolicy(primary="test-model"), PromptSpec("answer"))

    with pytest.raises(RuntimeInitializationError):
        await service.run(
            spec,
            "hi",
            principal=trusted_local_principal(tenant_id="t1"),
        )
