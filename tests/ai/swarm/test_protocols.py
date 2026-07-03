from linktools.ai.swarm.protocols import Task


def test_task_defaults():
    t = Task(task_id="a", payload={"x": 1})
    assert t.status == "pending"
    assert t.depends_on == ()
    assert t.claimed_by is None
    assert t.result is None
    assert t.error is None


def test_task_is_mutable_for_status_transitions():
    t = Task(task_id="a", payload=None)
    t.status = "claimed"
    t.claimed_by = "agent-1"
    assert t.status == "claimed"
    assert t.claimed_by == "agent-1"
