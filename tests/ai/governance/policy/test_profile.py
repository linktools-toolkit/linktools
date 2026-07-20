#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tests/ai/policy/test_profile.py

End-to-end proof that build_default_policy_engine actually wires a real
ToolRegistry into the rich rules (Permission/Risk/Approval): the engine built
from on-disk tool YAML denies the CRITICAL-destructive tool, allows the
LOW-read tool, escalates the HIGH-destructive tool to REQUIRE_APPROVAL, and
respects constructor overrides."""

import asyncio

from linktools.ai.governance.policy.engine import PolicyEngine
from linktools.ai.governance.policy.profile import build_default_policy_engine
from linktools.ai.governance.policy.rule import (
    PolicyDecisionKind,
    RiskLevel,
    SideEffectKind,
    ToolContext,
    ToolRequest,
)
from linktools.ai.catalog.parsing import SpecLoader
from linktools.ai.tool.catalog import ToolCatalog


def _write_tools(tmp_path) -> None:
    """Write three fixture tools under tmp_path/tools:

    - danger:     CRITICAL risk, destructive, execute+write
    - safe:       LOW risk,      read_only,  read
    - borderline: HIGH risk,     destructive, execute (risk <= default cap, but destructive)
    """
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "danger.yaml").write_text(
        "description: risky\n"
        "permissions: [execute, write]\n"
        "risk: CRITICAL\n"
        "side_effect: destructive\n"
        "approval: on_risk\n",
        encoding="utf-8",
    )
    (tools / "safe.yaml").write_text(
        "description: safe read\n"
        "permissions: [read]\n"
        "risk: LOW\n"
        "side_effect: read_only\n"
        "approval: never\n",
        encoding="utf-8",
    )
    (tools / "borderline.yaml").write_text(
        "description: high but destructive\n"
        "permissions: [execute]\n"
        "risk: HIGH\n"
        "side_effect: destructive\n"
        "approval: on_risk\n",
        encoding="utf-8",
    )


def _registry(tmp_path) -> ToolCatalog:
    return ToolCatalog.from_specloader(SpecLoader.from_filesystem(tmp_path / "tools"))


def _ctx() -> ToolContext:
    return ToolContext(run_id="r", session_id="s")


def _request(tool_name: str) -> ToolRequest:
    return ToolRequest(tool_name=tool_name, arguments={})


async def _run(tmp_path) -> None:
    _write_tools(tmp_path)
    registry = _registry(tmp_path)

    # 1. Default profile: max_risk=HIGH, approval for DESTRUCTIVE.
    engine = await build_default_policy_engine(registry)
    assert isinstance(engine, PolicyEngine)

    # 2. danger (CRITICAL > HIGH) -> DENY via RiskRule.
    decision = await engine.evaluate(_request("danger"), _ctx())
    assert decision.kind == PolicyDecisionKind.DENY
    assert decision.rule_id == "risk-rule"

    # 3. safe (LOW, read) -> ALLOW.
    decision = await engine.evaluate(_request("safe"), _ctx())
    assert decision.kind == PolicyDecisionKind.ALLOW

    # 4. borderline (HIGH, destructive) -> not denied by risk (HIGH <= HIGH),
    #    but ApprovalRule (require_side_effect=DESTRUCTIVE) -> REQUIRE_APPROVAL.
    decision = await engine.evaluate(_request("borderline"), _ctx())
    assert decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL
    assert decision.rule_id == "approval-rule"

    # 5. Override max_risk=CRITICAL -> danger no longer denied by risk
    #    (CRITICAL <= CRITICAL), but still destructive so ApprovalRule fires.
    engine_loose = await build_default_policy_engine(
        registry, max_risk=RiskLevel.CRITICAL
    )
    decision = await engine_loose.evaluate(_request("danger"), _ctx())
    assert decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL
    assert decision.rule_id == "approval-rule"

    # 6. Override approval_side_effect=NONE -> every side_effect (rank >= 0)
    #    escalates, so even the read-only "safe" tool now REQUIREs APPROVAL.
    engine_strict = await build_default_policy_engine(
        registry, approval_side_effect=SideEffectKind.NONE
    )
    decision = await engine_strict.evaluate(_request("safe"), _ctx())
    assert decision.kind == PolicyDecisionKind.REQUIRE_APPROVAL


def test_build_default_policy_engine(tmp_path):
    asyncio.run(_run(tmp_path))
