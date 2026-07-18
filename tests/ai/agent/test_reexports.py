#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""linktools.ai.agent already exports AgentSpec; this test proves
PromptSpec/ToolRef are now re-exported the same way (all three come from
the same agent/spec.py module), resolving to the exact same objects as the
deep submodule import."""


def test_prompt_spec_and_tool_ref_reexport_identity():
    from linktools.ai.agent import PromptSpec as PromptSpecShallow, ToolRef as ToolRefShallow
    from linktools.ai.agent.spec import PromptSpec as PromptSpecDeep, ToolRef as ToolRefDeep
    assert PromptSpecShallow is PromptSpecDeep
    assert ToolRefShallow is ToolRefDeep


def test_existing_agent_spec_export_still_works():
    """Regression guard: this task must not remove or break the existing export."""
    from linktools.ai.agent import AgentSpec
    assert AgentSpec is not None
