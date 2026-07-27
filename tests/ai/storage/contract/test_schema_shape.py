#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Schema-shape anchor: assert every table matches
linktools-ai/migrations/init_schema.sql (the enterprise-DB-conformant target).

Drives the refactor red→green per table. Inspects the two declarative bases
(Base = storage kernel, DomainBase = ai domain + commit-log rows).
"""
import pytest

from linktools.ai.storage.backends.sqlalchemy.models import Base as KernelBase
from linktools.ai.storage.sqlalchemy.models import Base as DomainBase

# Ensure the commit-log rows register on DomainBase.metadata.
import linktools.ai.run.persistence.sqlalchemy.commit_log  # noqa: F401
import linktools.ai.swarm.persistence.sqlalchemy_commit  # noqa: F401

TABLE_PREFIX = "ai_"

# (tablename_suffix, pk_col, required_cols, required_unique_names, required_index_names)
EXPECTED = [
    ("storage_objects", "id", {"id", "key", "key_hash", "created_at", "updated_at"},
     {"uk_key_hash"}, {"ix_key", "ix_updated_at", "ix_created_at"}),
    ("storage_object_versions", "id", {"id", "key_hash", "version", "created_at", "updated_at"},
     {"uk_key_hash_version"}, {"ix_updated_at", "ix_created_at"}),
    ("storage_object_idempotency", "id", {"id", "key_hash", "key", "created_at", "updated_at"},
     {"uk_key_hash"}, {"ix_key", "ix_updated_at", "ix_created_at"}),
    ("storage_object_revision", "id", {"id", "value", "created_at", "updated_at"},
     set(), {"ix_value", "ix_updated_at", "ix_created_at"}),
    ("storage_schema_version", "id", {"id", "component", "created_at", "updated_at"},
     {"uk_component"}, {"ix_updated_at", "ix_created_at"}),
    ("idempotency", "id",
     {"id", "idempotency_id", "scope_key_hash", "scope", "key", "error_message",
      "created_at", "updated_at"},
     {"uk_idempotency_id", "uk_scope_key_hash"},
     {"ix_scope", "ix_updated_at", "ix_created_at"}),
    ("runs", "id", {"id", "run_id", "data_json", "metadata_json", "created_at", "updated_at"},
     {"uk_run_id"},
     {"ix_root_run_id", "ix_parent_run_id", "ix_session_id", "ix_updated_at", "ix_created_at"}),
    ("run_checkpoints", "id", {"id", "checkpoint_id", "run_id", "created_at", "updated_at"},
     {"uk_checkpoint_id", "uk_run_id_sequence"}, {"ix_updated_at", "ix_created_at"}),
    ("run_checkpoint_counters", "id", {"id", "run_id", "created_at", "updated_at"},
     {"uk_run_id"}, {"ix_updated_at", "ix_created_at"}),
    ("run_definitions", "id", {"id", "run_id", "created_at", "updated_at"},
     {"uk_run_id"}, {"ix_updated_at", "ix_created_at"}),
    ("sessions", "id", {"id", "session_id", "created_at", "updated_at"},
     {"uk_session_id"}, {"ix_updated_at", "ix_created_at"}),
    ("session_messages", "id",
     {"id", "message_id", "session_id", "data_json", "commit_hash", "created_at",
      "updated_at"},
     {"uk_message_id", "uk_session_id_sequence", "uk_session_id_commit_hash_batch_index"},
     {"ix_updated_at", "ix_created_at"}),
    ("events", "id",
     {"id", "event_id", "stream_id", "data_json", "commit_hash", "created_at",
      "updated_at"},
     {"uk_event_id", "uk_stream_id_sequence", "uk_stream_id_commit_hash_event_type"},
     {"ix_run_id", "ix_updated_at", "ix_created_at"}),
    ("swarm_runs", "id", {"id", "swarm_run_id", "run_id", "created_at", "updated_at"},
     {"uk_swarm_run_id"}, {"ix_run_id", "ix_updated_at", "ix_created_at"}),
    ("swarm_tasks", "id",
     {"id", "task_id", "swarm_run_id", "data_json", "created_at", "updated_at"},
     {"uk_task_id"}, {"ix_swarm_run_id", "ix_updated_at", "ix_created_at"}),
    ("swarm_task_attempts", "id", {"id", "attempt_id", "task_id", "created_at", "updated_at"},
     {"uk_attempt_id"}, {"ix_task_id", "ix_run_id", "ix_updated_at", "ix_created_at"}),
    ("memories", "id", {"id", "memory_id", "tenant_id", "confidence", "created_at", "updated_at"},
     {"uk_memory_id"}, {"ix_tenant_id", "ix_updated_at", "ix_created_at"}),
    ("approvals", "id",
     {"id", "approval_id", "run_id", "tool_call_id", "data_json", "created_at",
      "updated_at"},
     {"uk_approval_id", "uk_run_id_tool_call_id"},
     {"ix_tenant_id", "ix_updated_at", "ix_created_at"}),
    ("jobs", "id", {"id", "job_id", "data_json", "created_at", "updated_at"},
     {"uk_job_id"},
     {"ix_status_created_at", "ix_tenant_id_status", "ix_updated_at", "ix_created_at"}),
    ("tasks", "id",
     {"id", "task_id", "job_id", "job_key_hash", "timeout_ms", "data_json", "created_at",
      "updated_at"},
     {"uk_task_id", "uk_job_key_hash"},
     {"ix_job_id_status", "ix_status_available_at", "ix_handler_status_available_at",
      "ix_lease_expires_at", "ix_updated_at", "ix_created_at"}),
    ("task_attempts", "id",
     {"id", "attempt_id", "task_id", "error_message", "data_json", "created_at",
      "updated_at"},
     {"uk_attempt_id", "uk_task_id_attempt"},
     {"ix_run_id", "ix_updated_at", "ix_created_at"}),
    ("task_transitions", "id", {"id", "job_id", "data_json", "created_at", "updated_at"},
     set(), {"ix_job_id", "ix_updated_at", "ix_created_at"}),
    ("task_signals", "id", {"id", "signal_id", "job_id", "data_json", "created_at", "updated_at"},
     {"uk_signal_id"}, {"ix_job_id_name", "ix_updated_at", "ix_created_at"}),
    ("eval_runs", "id", {"id", "eval_run_id", "data_json", "created_at", "updated_at"},
     {"uk_eval_run_id"}, {"ix_suite_id", "ix_updated_at", "ix_created_at"}),
    ("eval_results", "id",
     {"id", "result_id", "eval_run_id", "error_message", "data_json", "created_at",
      "updated_at"},
     {"uk_result_id"}, {"ix_eval_run_id", "ix_updated_at", "ix_created_at"}),
    ("artifact_records", "id",
     {"id", "artifact_id", "tenant_id", "content_hash", "data_json", "created_at",
      "updated_at"},
     {"uk_artifact_id"},
     {"ix_content_hash", "ix_updated_at", "ix_created_at"}),
    ("run_commit_log", "id",
     {"id", "commit_id", "commit_hash", "run_id", "created_at", "updated_at"},
     {"uk_commit_hash"}, {"ix_commit_id", "ix_run_id", "ix_updated_at", "ix_created_at"}),
    ("swarm_commit_log", "id",
     {"id", "commit_id", "commit_hash", "swarm_run_id", "created_at", "updated_at"},
     {"uk_commit_hash"},
     {"ix_commit_id", "ix_swarm_run_id", "ix_updated_at", "ix_created_at"}),
]


@pytest.fixture(scope="module")
def all_tables():
    tables = {}
    for metadata in (KernelBase.metadata, DomainBase.metadata):
        tables.update({t.name: t for t in metadata.sorted_tables})
    return tables


@pytest.mark.parametrize(
    "suffix,pk,cols,uniques,indexes", EXPECTED, ids=[e[0] for e in EXPECTED]
)
def test_table_matches_init_schema(all_tables, suffix, pk, cols, uniques, indexes):
    name = f"{TABLE_PREFIX}{suffix}"
    assert name in all_tables, f"table {name} missing from metadata"
    table = all_tables[name]
    assert [c.name for c in table.primary_key.columns] == [pk], f"{name}: pk is not {pk!r}"
    got_cols = {c.name for c in table.columns}
    assert cols <= got_cols, f"{name}: missing columns {sorted(cols - got_cols)}"
    got_unique = {
        c.name for c in table.constraints
        if c.__class__.__name__ == "UniqueConstraint" and c.name is not None
    }
    got_unique |= {i.name for i in table.indexes if i.unique and i.name is not None}
    assert uniques <= got_unique, f"{name}: missing uniques {sorted(uniques - got_unique)}"
    got_index = {i.name for i in table.indexes if i.name is not None}
    assert indexes <= got_index, f"{name}: missing indexes {sorted(indexes - got_index)}"
