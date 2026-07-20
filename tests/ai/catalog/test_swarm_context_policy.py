import pytest
from decimal import Decimal

from linktools.ai.errors import InvalidSpecError
from linktools.ai.swarm.codec import parse_swarm_spec
from linktools.ai.swarm.limits import SwarmLimits


def _payload():
    return {
        "agents": ["worker"],
        "coordinator": "coordinator",
        "context_policy": {
            "coordinator_reads_session": False,
            "worker_reads_session": True,
            "worker_reads_summary": False,
            "write_aggregate_to_session": False,
        },
    }


def test_swarm_registry_accepts_context_policy():
    spec = parse_swarm_spec("demo", _payload())
    assert spec.context_policy.coordinator_reads_session is False
    assert spec.context_policy.worker_reads_session is True
    assert spec.context_policy.worker_reads_summary is False
    assert spec.context_policy.write_aggregate_to_session is False


def test_swarm_registry_rejects_unknown_context_policy_field():
    payload = _payload()
    payload["context_policy"]["unknown"] = True
    with pytest.raises(InvalidSpecError):
        parse_swarm_spec("demo", payload)


def test_swarm_registry_rejects_invalid_context_policy_type():
    payload = _payload()
    payload["context_policy"] = 123
    with pytest.raises(InvalidSpecError):
        parse_swarm_spec("demo", payload)


def test_swarm_registry_keeps_unknown_top_level_fields_strict():
    payload = _payload()
    payload["unknown"] = True
    with pytest.raises(InvalidSpecError):
        parse_swarm_spec("demo", payload)


@pytest.mark.parametrize("field", ["max_rounds", "max_tasks", "max_concurrency"])
@pytest.mark.parametrize("value", [0, -1, False])
def test_swarm_registry_rejects_invalid_positive_limits(field, value):
    payload = _payload()
    payload["limits"] = {field: value}
    with pytest.raises(InvalidSpecError):
        parse_swarm_spec("demo", payload)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_swarm_registry_rejects_invalid_timeout(value):
    payload = _payload()
    payload["limits"] = {"timeout_seconds": value}
    with pytest.raises(InvalidSpecError):
        parse_swarm_spec("demo", payload)


def test_swarm_registry_rejects_invalid_cost_values():
    for value in [-1, "1.2", float("nan"), float("inf")]:
        payload = _payload()
        payload["limits"] = {"max_total_cost": value}
        with pytest.raises(InvalidSpecError):
            parse_swarm_spec("demo", payload)


def test_swarm_limits_programmatic_validation():
    with pytest.raises(ValueError):
        SwarmLimits(0, 1, 0, 0, 1, None, None, None)
    with pytest.raises(ValueError):
        SwarmLimits(1, 1, 0, 0, 1, None, Decimal("-1"), None)
    with pytest.raises(ValueError):
        SwarmLimits(1, 1, 0, 0, 1, None, None, float("inf"))
