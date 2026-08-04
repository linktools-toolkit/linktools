"""Deterministic identifiers used by task_graph persistence."""

from hashlib import sha256


def task_execution_id(plan_id: str, node_id: str) -> str:
    digest = sha256(
        b"task-execution-v1\0"
        + plan_id.encode("utf-8")
        + b"\0"
        + node_id.encode("utf-8")
    ).hexdigest()
    return f"tg-task-{digest}"


def child_run_id(parent_run_id: str, node_id: str) -> str:
    digest = sha256(
        b"task-graph-child-v1\0"
        + parent_run_id.encode("utf-8")
        + b"\0"
        + node_id.encode("utf-8")
    ).hexdigest()
    return f"tg-child-{digest}"


__all__ = ["child_run_id", "task_execution_id"]
