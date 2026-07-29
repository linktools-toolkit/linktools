#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""The agent package exposes its common agent-domain declarations."""


def test_prompt_spec_reexport_identity():
    from linktools.ai.agent import PromptSpec as PromptSpecShallow
    from linktools.ai.agent.spec import PromptSpec as PromptSpecDeep

    assert PromptSpecShallow is PromptSpecDeep


def test_existing_agent_spec_export_still_works():
    """Regression guard: this task must not remove or break the existing export."""
    from linktools.ai.agent import AgentSpec
    assert AgentSpec is not None
