import asyncio

import pytest

from linktools.ai.errors import InvalidSpecError
from linktools.ai.registry.parser import SpecLoader
from linktools.ai.registry.tool import ToolRegistry
from linktools.ai.tool.policy import (
    EffectiveToolPolicy,
    IdempotencyStrategy,
    ResolvedToolPolicy,
    finalize_policy,
    merge_policies,
)

BUSINESS = IdempotencyStrategy.BUSINESS_KEY
EXACT = IdempotencyStrategy.EXACT_CALL


# --- ResolvedToolPolicy (declaration layer) ---


def test_resolved_business_key_requires_key_field():
    with pytest.raises(ValueError, match="idempotency_key_field"):
        ResolvedToolPolicy(idempotent=True, idempotency_strategy=BUSINESS)


def test_resolved_business_key_rejects_blank_key_field():
    with pytest.raises(ValueError, match="idempotency_key_field"):
        ResolvedToolPolicy(
            idempotent=True, idempotency_strategy=BUSINESS, idempotency_key_field="   "
        )


def test_resolved_business_key_rejects_idempotent_false():
    with pytest.raises(ValueError, match="idempotent=true"):
        ResolvedToolPolicy(
            idempotent=False, idempotency_strategy=BUSINESS, idempotency_key_field="x"
        )


def test_resolved_business_key_allows_idempotent_none_with_field():
    # Tri-state declaration: a layer may declare business_key + field and leave
    # idempotent for another layer to set true; finalize/Effective is the gate.
    p = ResolvedToolPolicy(idempotency_strategy=BUSINESS, idempotency_key_field="x")
    assert p.idempotent is None


def test_resolved_business_key_accepts_complete_declaration():
    p = ResolvedToolPolicy(
        idempotent=True, idempotency_strategy=BUSINESS, idempotency_key_field="ext_id"
    )
    assert p.idempotency_strategy is BUSINESS


def test_resolved_key_field_requires_business_key():
    with pytest.raises(ValueError, match="business_key"):
        ResolvedToolPolicy(idempotency_strategy=EXACT, idempotency_key_field="x")


def test_resolved_idempotent_false_with_no_strategy_is_allowed():
    p = ResolvedToolPolicy(idempotent=False)
    assert p.idempotent is False


def test_resolved_idempotent_false_with_explicit_exact_call_is_rejected():
    # Declaring a strategy while saying idempotent=false is contradictory even
    # for the default strategy -- exact_call is the effective-layer fallback, not
    # a valid declaration to pair with idempotent=false.
    with pytest.raises(ValueError, match="idempotent=true"):
        ResolvedToolPolicy(idempotent=False, idempotency_strategy=EXACT)


def test_resolved_strips_key_field_whitespace():
    p = ResolvedToolPolicy(
        idempotent=True, idempotency_strategy=BUSINESS, idempotency_key_field="  ext_id  "
    )
    assert p.idempotency_key_field == "ext_id"


# --- EffectiveToolPolicy (finalized layer) ---


def test_effective_business_key_requires_key_field():
    with pytest.raises(ValueError, match="idempotency_key_field"):
        EffectiveToolPolicy(idempotent=True, idempotency_strategy=BUSINESS)


def test_effective_business_key_rejects_idempotent_false():
    with pytest.raises(ValueError, match="idempotent=true"):
        EffectiveToolPolicy(
            idempotent=False, idempotency_strategy=BUSINESS, idempotency_key_field="x"
        )


def test_effective_business_key_requires_explicit_idempotent_true():
    # At the effective layer business_key must collapse to idempotent=True, not None.
    with pytest.raises(ValueError, match="idempotent=true"):
        EffectiveToolPolicy(idempotency_strategy=BUSINESS, idempotency_key_field="x")


def test_effective_default_non_idempotent_is_allowed():
    # The normal finalized non-idempotent policy: idempotent=False + EXACT_CALL
    # (finalize_policy's fallback) must remain valid.
    assert EffectiveToolPolicy().idempotent is False
    assert EffectiveToolPolicy(idempotent=False).idempotency_strategy is EXACT


def test_effective_business_key_accepts_complete_policy():
    p = EffectiveToolPolicy(
        idempotent=True, idempotency_strategy=BUSINESS, idempotency_key_field="ext_id"
    )
    assert p.idempotency_strategy is BUSINESS


# --- merge / finalize (the dynamic-provider path) ---


def test_dynamic_provider_cannot_construct_invalid_business_key_policy():
    # A dynamic ToolPolicyProvider cannot hand back a business_key policy without
    # a key_field: the ResolvedToolPolicy it would return fails at construction,
    # before the provider's result ever reaches the executor or idempotency store.
    with pytest.raises(ValueError, match="idempotency_key_field"):
        ResolvedToolPolicy(idempotent=True, idempotency_strategy=BUSINESS)


def test_merge_preserves_valid_business_key_provider_layer():
    provider = ResolvedToolPolicy(
        idempotent=True, idempotency_strategy=BUSINESS, idempotency_key_field="ext_id"
    )
    merged = merge_policies(None, None, provider)
    assert merged.idempotency_strategy is BUSINESS
    assert merged.idempotency_key_field == "ext_id"


def test_finalize_rejects_business_key_without_key_field():
    resolved = ResolvedToolPolicy(
        idempotent=True, idempotency_strategy=BUSINESS, idempotency_key_field="ext_id"
    )
    effective = finalize_policy(resolved)
    assert effective.idempotency_strategy is BUSINESS
    # Forcing the invalid combination through finalize still fails at Effective.
    with pytest.raises(ValueError, match="idempotency_key_field"):
        EffectiveToolPolicy(
            idempotent=effective.idempotent,
            idempotency_strategy=BUSINESS,
        )


# --- Registry (YAML load-time gate) ---


def _load_tool_spec(tmp_path, body: str):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "decl.yaml").write_text(body, encoding="utf-8")
    registry = ToolRegistry(SpecLoader.from_filesystem(tools))

    async def run():
        return await registry.get("decl")

    return asyncio.run(run())


def test_registry_rejects_business_key_without_key_field(tmp_path):
    with pytest.raises(InvalidSpecError, match="idempotency_key_field"):
        _load_tool_spec(
            tmp_path,
            "idempotent: true\nidempotency_strategy: business_key\n",
        )


def test_registry_rejects_business_key_with_idempotent_false(tmp_path):
    with pytest.raises(InvalidSpecError, match="idempotent=true"):
        _load_tool_spec(
            tmp_path,
            "idempotent: false\nidempotency_strategy: business_key\n"
            "idempotency_key_field: ext_id\n",
        )


def test_registry_accepts_valid_business_key_declaration(tmp_path):
    spec = _load_tool_spec(
        tmp_path,
        "idempotent: true\nidempotency_strategy: business_key\n"
        "idempotency_key_field: ext_id\n",
    )
    assert spec.idempotency_strategy == "business_key"
    assert spec.idempotency_key_field == "ext_id"
