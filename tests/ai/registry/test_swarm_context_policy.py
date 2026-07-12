import pytest

from linktools.ai.errors import InvalidSpecError
from linktools.ai.registry.swarm import parse_swarm_spec


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
