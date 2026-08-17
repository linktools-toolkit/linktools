#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned SQL metadata and plan-to-table selection."""

from typing import TYPE_CHECKING

from ...storage import (
    sql_audit_columns,
    sql_digest,
    sql_id_column,
    sql_integer_id,
    sql_query_index,
    sql_table_options,
    sql_text_key,
    sql_unique,
)
from ._plan import RuntimeDomain, RuntimeRetentionMode, RuntimeStatePlan

if TYPE_CHECKING:
    from sqlalchemy import MetaData, Table


_STEP_DOMAINS = frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY})

def required_runtime_sql_tables(plan: RuntimeStatePlan) -> frozenset[str]:
    names: set[str] = set()
    for domain in RuntimeDomain:
        if plan.route(domain).retention is RuntimeRetentionMode.DURABLE:
            names.update(_RUNTIME_DOMAIN_TABLES[domain])
    return frozenset(names)


def _runtime_sql_tables_for_domains(domains: frozenset[RuntimeDomain]) -> frozenset[str]:
    names: set[str] = set()
    for domain in domains:
        names.update(_RUNTIME_DOMAIN_TABLES[domain])
    return frozenset(names)


def build_runtime_sql_metadata(
    plan: "RuntimeStatePlan | frozenset[RuntimeDomain]",
    *,
    metadata: "MetaData | None" = None,
) -> "MetaData":
    from sqlalchemy import Boolean, Column, DateTime, JSON, MetaData, Table, Text

    if metadata is None:
        metadata = MetaData()
    tables = (
        _runtime_sql_tables_for_domains(plan)
        if isinstance(plan, frozenset)
        else required_runtime_sql_tables(plan)
    )
    common = {
        "namespace_digest": (sql_digest(), "Namespace SHA-256 digest"),
        "tenant_id": (sql_text_key(128), "Tenant identifier"),
    }

    def table(
        name: str,
        fields: dict[str, tuple[object, str]],
        unique: tuple[str, ...],
        *,
        table_comment: str,
        nullable: frozenset[str] = frozenset(),
        extra_unique: tuple[tuple[str, ...], ...] = (),
    ) -> "Table | None":
        if name in metadata.tables:
            return metadata.tables[name]
        columns = [sql_id_column()]
        for field, (value, field_comment) in fields.items():
            columns.append(
                Column(
                    field,
                    value,
                    nullable=field in nullable,
                    comment=field_comment,
                )
            )
        columns.extend(sql_audit_columns())
        owner_table = Table(
            name,
            metadata,
            *columns,
            comment=table_comment,
            **sql_table_options(),
        )
        sql_unique(owner_table, *unique)
        for constraint in extra_unique:
            sql_unique(owner_table, *constraint)
        sql_query_index(owner_table, "updated_at")
        sql_query_index(owner_table, "created_at")
        return owner_table

    if "ai_runtime_sessions" in tables:
        table(
            "ai_runtime_sessions",
            {
                **common,
                "session_id": (sql_text_key(128), "Session identifier"),
                "owner_principal_id": (sql_text_key(), "Owner principal identifier"),
                "binding_digest": (sql_digest(), "Binding definition SHA-256 digest"),
                "status": (sql_text_key(64), "Status"),
                "revision": (sql_integer_id(), "Resource revision"),
                "resource_generation": (sql_integer_id(), "Resource generation"),
                "cwd": (Text(), "Working directory"),
                "metadata": (JSON(), "Extended metadata"),
                "continuation_step_run_id": (sql_text_key(), "Continuation Step run identifier"),
                "active_execution_id": (sql_text_key(128), "Durably admitted execution identifier"),
                "closed_at": (DateTime(timezone=True), "Close time"),
            },
            ("namespace_digest", "tenant_id", "session_id"),
            table_comment="Runtime sessions",
            nullable=frozenset({"cwd", "continuation_step_run_id", "active_execution_id", "closed_at"}),
        )
    if "ai_runtime_executions" in tables:
        table(
            "ai_runtime_executions",
            {
                **common,
                "execution_id": (sql_text_key(128), "Execution identifier"),
                "session_id": (sql_text_key(128), "Session identifier"),
                "binding_digest": (sql_digest(), "Binding definition SHA-256 digest"),
                "parent_execution_id": (sql_text_key(128), "Parent execution identifier"),
                "root_execution_id": (sql_text_key(128), "Root execution identifier"),
                "source_execution_id": (sql_text_key(128), "Source execution identifier"),
                "base_execution_id": (sql_text_key(128), "Base execution identifier"),
                "lineage_kind": (sql_text_key(64), "Execution lineage kind"),
                "status": (sql_text_key(64), "Status"),
                "revision": (sql_integer_id(), "Resource revision"),
                "event_sequence": (sql_integer_id(), "Event sequence number"),
                "agent_run_sequence": (sql_integer_id(), "Agent run sequence number"),
                "error_code": (sql_text_key(128), "Error code"),
                "safe_error_details": (JSON(), "Safe error details"),
                "memory_scope": (sql_text_key(), "Memory scope"),
                "conversation_step_run_id": (sql_text_key(), "Conversation Step run identifier"),
                "output_schema_id": (sql_text_key(), "Output schema identifier"),
                "output_schema_revision": (sql_integer_id(), "Output schema revision"),
                "output_schema_fingerprint": (sql_digest(), "Output schema fingerprint"),
                "result_store_id": (sql_text_key(128), "Result ObjectStore identifier"),
                "result_key": (sql_text_key(255), "Result object key"),
                "result_digest": (sql_digest(), "Result SHA-256 digest"),
                "result_size": (sql_integer_id(), "Result size in bytes"),
                "stop_reason": (Text(), "Stop reason"),
                "model_requests": (sql_integer_id(), "Model request count"),
                "tool_calls": (sql_integer_id(), "Tool call count"),
                "input_tokens": (sql_integer_id(), "Input token count"),
                "output_tokens": (sql_integer_id(), "Output token count"),
                "cache_read_tokens": (sql_integer_id(), "Cache read token count"),
                "cache_write_tokens": (sql_integer_id(), "Cache write token count"),
                "result_created_at": (DateTime(timezone=True), "Result creation time"),
            },
            ("namespace_digest", "tenant_id", "execution_id"),
            table_comment="Runtime executions",
            nullable=frozenset(
                {
                    "session_id",
                    "parent_execution_id",
                    "source_execution_id",
                    "base_execution_id",
                    "error_code",
                    "memory_scope",
                    "conversation_step_run_id",
                    "output_schema_id",
                    "output_schema_revision",
                    "output_schema_fingerprint",
                    "result_store_id",
                    "result_key",
                    "result_digest",
                    "result_size",
                    "stop_reason",
                    "model_requests",
                    "tool_calls",
                    "input_tokens",
                    "output_tokens",
                    "cache_read_tokens",
                    "cache_write_tokens",
                    "result_created_at",
                }
            ),
        )
    if "ai_runtime_idempotency" in tables:
        table(
            "ai_runtime_idempotency",
            {
                **common,
                "runtime_domain": (sql_text_key(64), "Runtime domain"),
                "scope": (sql_text_key(128), "Idempotency scope"),
                "idempotency_key_digest": (sql_digest(), "Idempotency key SHA-256 digest"),
                "identity_digest": (sql_digest(), "Logical identity SHA-256 digest"),
                "request_digest": (sql_digest(), "Request SHA-256 digest"),
                "resource_id": (sql_text_key(128), "Resource identifier"),
                "status": (sql_text_key(64), "Status"),
                "result_digest": (sql_digest(), "Result SHA-256 digest"),
                "error_code": (sql_text_key(128), "Error code"),
            },
            ("namespace_digest", "tenant_id", "identity_digest"),
            table_comment="Runtime idempotency records",
            nullable=frozenset({"result_digest", "error_code"}),
        )
    if "ai_runtime_events" in tables:
        events_table = table(
            "ai_runtime_events",
            {
                **common,
                "execution_id": (sql_text_key(128), "Execution identifier"),
                "sequence": (sql_integer_id(), "Sequence number"),
                "identity_digest": (sql_digest(), "Logical identity SHA-256 digest"),
                "kind": (sql_text_key(128), "Kind"),
                "payload": (JSON(), "Event payload"),
            },
            ("namespace_digest", "tenant_id", "identity_digest"),
            table_comment="Runtime execution events",
        )
        if events_table is not None:
            sql_query_index(
                events_table,
                "namespace_digest",
                "tenant_id",
                "execution_id",
                mysql_length=128,
            )
    if "ai_runtime_task_graphs" in tables:
        table(
            "ai_runtime_task_graphs",
            {
                **common,
                "graph_id": (sql_text_key(128), "Task graph identifier"),
                "status": (sql_text_key(64), "Status"),
                "revision": (sql_integer_id(), "Resource revision"),
            },
            ("namespace_digest", "tenant_id", "graph_id"),
            table_comment="Runtime task graphs",
        )
    if "ai_runtime_task_nodes" in tables:
        nodes_table = table(
            "ai_runtime_task_nodes",
            {
                **common,
                "graph_id": (sql_text_key(128), "Task graph identifier"),
                "node_id": (sql_text_key(128), "Task node identifier"),
                "identity_digest": (sql_digest(), "Logical identity SHA-256 digest"),
                "dependencies": (JSON(), "Dependency node set"),
                "input": (JSON(), "Input data"),
                "budget_cost": (sql_integer_id(), "Budget cost"),
                "status": (sql_text_key(64), "Status"),
                "revision": (sql_integer_id(), "Resource revision"),
                "owner": (sql_text_key(), "Lease owner"),
                "fence": (sql_integer_id(), "Lease fence value"),
                "lease_expires_at": (DateTime(timezone=True), "Lease expiration time"),
                "execution_id": (sql_text_key(128), "Execution identifier"),
                "result_digest": (sql_digest(), "Result SHA-256 digest"),
                "error_code": (sql_text_key(128), "Error code"),
                "error_digest": (sql_digest(), "Error SHA-256 digest"),
            },
            ("namespace_digest", "tenant_id", "identity_digest"),
            table_comment="Runtime task nodes",
            nullable=frozenset(
                {
                    "input",
                    "budget_cost",
                    "owner",
                    "lease_expires_at",
                    "execution_id",
                    "result_digest",
                    "error_code",
                    "error_digest",
                }
            ),
        )
        if nodes_table is not None:
            sql_query_index(
                nodes_table,
                "namespace_digest",
                "tenant_id",
                "graph_id",
                mysql_length=128,
            )
    if "ai_runtime_evaluations" in tables:
        table(
            "ai_runtime_evaluations",
            {
                **common,
                "evaluation_id": (sql_text_key(128), "Evaluation identifier"),
                "execution_id": (sql_text_key(128), "Execution identifier"),
                "dataset_id": (sql_text_key(), "Dataset identifier"),
                "dataset_revision": (sql_integer_id(), "Dataset revision"),
                "evaluator_id": (sql_text_key(), "Evaluator identifier"),
                "evaluator_revision": (sql_integer_id(), "Evaluator revision"),
                "binding_digest": (sql_digest(), "Binding definition SHA-256 digest"),
                "output_schema_fingerprint": (sql_digest(), "Output schema fingerprint"),
                "artifact_digest": (sql_digest(), "Artifact SHA-256 digest"),
                "status": (sql_text_key(64), "Status"),
                "revision": (sql_integer_id(), "Resource revision"),
                "metrics": (JSON(), "Evaluation metrics"),
            },
            ("namespace_digest", "tenant_id", "evaluation_id"),
            table_comment="Runtime evaluations",
            nullable=frozenset({"artifact_digest"}),
        )
    if "ai_runtime_memories" in tables:
        table(
            "ai_runtime_memories",
            {
                **common,
                "memory_id": (sql_text_key(128), "Memory identifier"),
                "memory_scope_digest": (sql_digest(), "Memory scope SHA-256 digest"),
                "content_store_id": (sql_text_key(128), "Content ObjectStore identifier"),
                "content_key": (sql_text_key(255), "Content object key"),
                "content_digest": (sql_digest(), "Content SHA-256 digest"),
                "content_size": (sql_integer_id(), "Content size in bytes"),
                "metadata": (JSON(), "Extended metadata"),
                "revision": (sql_integer_id(), "Resource revision"),
            },
            ("namespace_digest", "tenant_id", "memory_id"),
            table_comment="Runtime memories",
        )
    if "ai_runtime_artifacts" in tables:
        table(
            "ai_runtime_artifacts",
            {
                **common,
                "artifact_id": (sql_text_key(128), "Artifact identifier"),
                "execution_id": (sql_text_key(128), "Execution identifier"),
                "producer": (sql_text_key(), "Artifact producer"),
                "media_type": (sql_text_key(), "Media type"),
                "content_store_id": (sql_text_key(128), "Content ObjectStore identifier"),
                "content_key": (sql_text_key(255), "Content object key"),
                "content_digest": (sql_digest(), "Content SHA-256 digest"),
                "content_size": (sql_integer_id(), "Content size in bytes"),
            },
            ("namespace_digest", "tenant_id", "artifact_id"),
            table_comment="Runtime artifacts",
        )
    if "ai_runtime_approvals" in tables:
        table(
            "ai_runtime_approvals",
            {
                **common,
                "approval_id": (sql_text_key(128), "Approval identifier"),
                "execution_id": (sql_text_key(128), "Execution identifier"),
                "operation_id": (sql_text_key(128), "Operation identifier"),
                "status": (sql_text_key(64), "Status"),
                "idempotency_key_digest": (sql_digest(), "Idempotency key SHA-256 digest"),
                "decision": (sql_text_key(64), "Approval decision"),
                "decided_by": (sql_text_key(), "Decision principal"),
                "decision_digest": (sql_digest(), "Approval decision SHA-256 digest"),
                "decided_at": (DateTime(timezone=True), "Decision time"),
            },
            ("namespace_digest", "tenant_id", "approval_id"),
            table_comment="Runtime approvals",
            nullable=frozenset(
                {"idempotency_key_digest", "decision", "decided_by", "decision_digest", "decided_at"}
            ),
        )
    if "ai_runtime_external_calls" in tables:
        table(
            "ai_runtime_external_calls",
            {
                **common,
                "call_id": (sql_text_key(128), "External call identifier"),
                "execution_id": (sql_text_key(128), "Execution identifier"),
                "operation_id": (sql_text_key(128), "Operation identifier"),
                "status": (sql_text_key(64), "Status"),
                "idempotency_key_digest": (sql_digest(), "Idempotency key SHA-256 digest"),
                "result_store_id": (sql_text_key(128), "Result ObjectStore identifier"),
                "result_key": (sql_text_key(255), "Result object key"),
                "result_digest": (sql_digest(), "Result SHA-256 digest"),
                "result_size": (sql_integer_id(), "Result size in bytes"),
                "payload_digest": (sql_digest(), "Payload SHA-256 digest"),
                "supplied_at": (DateTime(timezone=True), "Result supplied time"),
            },
            ("namespace_digest", "tenant_id", "call_id"),
            table_comment="Runtime external calls",
            nullable=frozenset(
                {
                    "idempotency_key_digest",
                    "result_store_id",
                    "result_key",
                    "result_digest",
                    "result_size",
                    "payload_digest",
                    "supplied_at",
                }
            ),
        )
    if "ai_runtime_recovery_checkpoints" in tables:
        table(
            "ai_runtime_recovery_checkpoints",
            {
                **common,
                "execution_id": (sql_text_key(128), "Execution identifier"),
                "step_run_id": (sql_text_key(), "Step run identifier"),
                "agent_run_sequence": (sql_integer_id(), "Agent run sequence number"),
                "state": (sql_text_key(64), "State data"),
                "handoff_phase": (sql_text_key(64), "Handoff phase"),
                "input": (JSON(), "Input data"),
                "terminal_handoff": (JSON(), "Terminal handoff data"),
                "handoff_contract_digest": (sql_digest(), "Handoff contract SHA-256 digest"),
                "pending_operation_id": (sql_text_key(128), "Pending operation identifier"),
                "revision": (sql_integer_id(), "Resource revision"),
            },
            ("namespace_digest", "tenant_id", "execution_id"),
            table_comment="Runtime recovery checkpoints",
            nullable=frozenset({"terminal_handoff", "handoff_contract_digest", "pending_operation_id"}),
        )
    if "ai_runtime_tool_operations" in tables:
        table(
            "ai_runtime_tool_operations",
            {
                **common,
                "tool_operation_id": (sql_text_key(128), "Tool operation identifier"),
                "step_run_id": (sql_text_key(), "Step run identifier"),
                "tool_call_id": (sql_text_key(), "Tool call identifier"),
                "call_digest": (sql_digest(), "Tool call identity SHA-256 digest"),
                "idempotency_key_digest": (sql_digest(), "Idempotency key SHA-256 digest"),
                "tool_name": (sql_text_key(), "Tool name"),
                "arguments_digest": (sql_digest(), "Tool arguments SHA-256 digest"),
                "binding_fingerprint": (sql_digest(), "Binding fingerprint"),
                "replay_safe": (Boolean(), "Replay safety flag"),
                "status": (sql_text_key(64), "Status"),
                "owner": (sql_text_key(), "Lease owner"),
                "fence": (sql_integer_id(), "Lease fence value"),
                "lease_expires_at": (DateTime(timezone=True), "Lease expiration time"),
                "result_store_id": (sql_text_key(128), "Result ObjectStore identifier"),
                "result_key": (sql_text_key(255), "Result object key"),
                "result_digest": (sql_digest(), "Result SHA-256 digest"),
                "result_size": (sql_integer_id(), "Result size in bytes"),
                "error_code": (sql_text_key(128), "Error code"),
            },
            ("namespace_digest", "tenant_id", "tool_operation_id"),
            table_comment="Runtime tool operations",
            nullable=frozenset(
                {
                    "owner",
                    "lease_expires_at",
                    "result_store_id",
                    "result_key",
                    "result_digest",
                    "result_size",
                    "error_code",
                }
            ),
            extra_unique=(("namespace_digest", "tenant_id", "call_digest"),),
        )
    if "ai_runtime_operation_counters" in tables:
        table(
            "ai_runtime_operation_counters",
            {
                **common,
                "runtime_domain": (sql_text_key(64), "Runtime domain"),
                "resource_kind": (sql_text_key(128), "Resource kind"),
                "resource_id": (sql_text_key(128), "Resource identifier"),
                "stream_digest": (sql_digest(), "Operation stream SHA-256 digest"),
                "last_sequence": (sql_integer_id(), "Last allocated sequence"),
            },
            ("namespace_digest", "stream_digest"),
            table_comment="Runtime operation sequence counters",
        )
    if "ai_runtime_operations" in tables:
        table(
            "ai_runtime_operations",
            {
                **common,
                "runtime_domain": (sql_text_key(64), "Runtime domain"),
                "resource_kind": (sql_text_key(128), "Resource kind"),
                "resource_id": (sql_text_key(128), "Resource identifier"),
                "stream_digest": (sql_digest(), "Operation stream SHA-256 digest"),
                "operation_kind": (sql_text_key(128), "Operation kind"),
                "operation_id": (sql_text_key(128), "Operation identifier"),
                "operation_digest": (sql_digest(), "Operation identity SHA-256 digest"),
                "sequence": (sql_integer_id(), "Sequence number"),
                "status": (sql_text_key(64), "Status"),
                "execution_id": (sql_text_key(128), "Execution identifier"),
                "request_digest": (sql_digest(), "Request SHA-256 digest"),
                "result_ref": (Text(), "Operation result reference"),
                "result_digest": (sql_digest(), "Result SHA-256 digest"),
                "error_code": (sql_text_key(128), "Error code"),
                "compactable": (Boolean(), "Compaction eligibility flag"),
            },
            ("namespace_digest", "tenant_id", "operation_digest"),
            table_comment="Runtime operation ledger",
            nullable=frozenset({"execution_id", "result_ref", "result_digest", "error_code"}),
            extra_unique=(("namespace_digest", "stream_digest", "sequence"),),
        )
    return metadata


def _build_step_tables(metadata: "MetaData", names: set[str]) -> None:
    from sqlalchemy import Column, DateTime, JSON, Table, Text

    common = {
        "namespace_digest": (sql_digest(), "Namespace SHA-256 digest"),
        "tenant_id": (sql_text_key(128), "Tenant identifier"),
    }

    def table(
        name: str,
        fields: dict[str, tuple[object, str]],
        unique: tuple[str, ...],
        *,
        table_comment: str,
        nullable: frozenset[str] = frozenset(),
    ) -> "Table | None":
        if name in metadata.tables:
            return None
        columns = [sql_id_column()]
        for field, (value, field_comment) in fields.items():
            columns.append(
                Column(
                    field,
                    value,
                    nullable=field in nullable,
                    comment=field_comment,
                )
            )
        columns.extend(sql_audit_columns())
        owner_table = Table(
            name,
            metadata,
            *columns,
            comment=table_comment,
            **sql_table_options(),
        )
        sql_unique(owner_table, *unique)
        sql_query_index(owner_table, "updated_at")
        sql_query_index(owner_table, "created_at")
        return owner_table

    if "ai_step_runs" in names:
        runs_table = table(
            "ai_step_runs",
            {
                **common,
                "runtime_domain": (sql_text_key(64), "Runtime domain"),
                "run_id": (sql_text_key(), "Step run identifier"),
                "identity_digest": (sql_digest(), "Logical identity SHA-256 digest"),
                "conversation_id": (sql_text_key(), "Conversation identifier"),
                "parent_run_id": (sql_text_key(), "Parent Step run identifier"),
                "agent_name": (sql_text_key(), "Agent name"),
                "metadata": (JSON(), "Extended metadata"),
                "last_event_index": (sql_integer_id(), "Last event index"),
                "last_snapshot_index": (sql_integer_id(), "Last snapshot index"),
                "last_effect_index": (sql_integer_id(), "Last effect index"),
                "started_at": (DateTime(timezone=True), "Start time"),
            },
            ("namespace_digest", "tenant_id", "identity_digest"),
            table_comment="Step runs",
            nullable=frozenset({"conversation_id", "parent_run_id", "agent_name"}),
        )
        if runs_table is not None:
            sql_query_index(
                runs_table,
                "namespace_digest",
                "tenant_id",
                "conversation_id",
                mysql_length=128,
            )
            sql_query_index(
                runs_table,
                "namespace_digest",
                "tenant_id",
                "parent_run_id",
                mysql_length=128,
            )
    if "ai_step_events" in names:
        events_table = table(
            "ai_step_events",
            {
                **common,
                "run_id": (sql_text_key(), "Step run identifier"),
                "event_index": (sql_integer_id(), "Event index"),
                "identity_digest": (sql_digest(), "Logical identity SHA-256 digest"),
                "kind": (sql_text_key(), "Kind"),
                "step_index": (sql_integer_id(), "Step index"),
                "timestamp": (DateTime(timezone=True), "Event timestamp"),
                "conversation_id": (sql_text_key(), "Conversation identifier"),
                "parent_run_id": (sql_text_key(), "Parent Step run identifier"),
                "agent_name": (sql_text_key(), "Agent name"),
                "tool_call_id": (sql_text_key(), "Tool call identifier"),
                "tool_name": (sql_text_key(), "Tool name"),
                "error": (Text(), "Error message"),
                "metadata": (JSON(), "Extended metadata"),
            },
            ("namespace_digest", "tenant_id", "identity_digest"),
            table_comment="Step events",
            nullable=frozenset(
                {"conversation_id", "parent_run_id", "agent_name", "tool_call_id", "tool_name", "error"}
            ),
        )
        if events_table is not None:
            sql_query_index(
                events_table,
                "namespace_digest",
                "tenant_id",
                "run_id",
                mysql_length=128,
            )
    if "ai_step_snapshots" in names:
        snapshots_table = table(
            "ai_step_snapshots",
            {
                **common,
                "runtime_domain": (sql_text_key(64), "Runtime domain"),
                "run_id": (sql_text_key(), "Step run identifier"),
                "snapshot_index": (sql_integer_id(), "Snapshot index"),
                "identity_digest": (sql_digest(), "Logical identity SHA-256 digest"),
                "step_index": (sql_integer_id(), "Step index"),
                "state": (sql_text_key(64), "State data"),
                "conversation_id": (sql_text_key(), "Conversation identifier"),
                "parent_run_id": (sql_text_key(), "Parent Step run identifier"),
                "agent_name": (sql_text_key(), "Agent name"),
                "timestamp": (DateTime(timezone=True), "Event timestamp"),
                "media_store_id": (sql_text_key(128), "Media ObjectStore identifier"),
                "messages": (JSON(), "Message snapshot"),
            },
            ("namespace_digest", "tenant_id", "identity_digest"),
            table_comment="Step snapshots",
            nullable=frozenset({"conversation_id", "parent_run_id", "agent_name"}),
        )
        if snapshots_table is not None:
            sql_query_index(
                snapshots_table,
                "namespace_digest",
                "tenant_id",
                "run_id",
                mysql_length=128,
            )
    if "ai_step_effects" in names:
        effects = table(
            "ai_step_effects",
            {
                **common,
                "run_id": (sql_text_key(), "Step run identifier"),
                "effect_index": (sql_integer_id(), "Effect index"),
                "identity_digest": (sql_digest(), "Logical identity SHA-256 digest"),
                "tool_call_id": (sql_text_key(), "Tool call identifier"),
                "tool_name": (sql_text_key(), "Tool name"),
                "status": (sql_text_key(64), "Status"),
                "started_at": (DateTime(timezone=True), "Start time"),
                "ended_at": (DateTime(timezone=True), "End time"),
                "idempotency_key": (sql_text_key(), "Idempotency key"),
                "effect_summary": (Text(), "Effect summary"),
            },
            ("namespace_digest", "tenant_id", "identity_digest"),
            table_comment="Step effects",
            nullable=frozenset({"ended_at", "idempotency_key", "effect_summary"}),
        )
        if effects is not None:
            sql_query_index(effects, "namespace_digest", "tenant_id", "run_id", mysql_length=128)


def build_step_sql_metadata(
    runtime_domain: RuntimeDomain,
    *,
    metadata: "MetaData | None" = None,
) -> "MetaData":
    if runtime_domain not in _STEP_DOMAINS:
        raise ValueError("Step archive owner is invalid")
    from sqlalchemy import MetaData

    if metadata is None:
        metadata = MetaData()
    names = {"ai_step_runs", "ai_step_snapshots"}
    if runtime_domain is RuntimeDomain.EXECUTION:
        names.add("ai_step_events")
    if runtime_domain is RuntimeDomain.RECOVERY:
        names.add("ai_step_effects")
    _build_step_tables(metadata, names)
    return metadata


_RUNTIME_DOMAIN_TABLES = {
    RuntimeDomain.CONVERSATION: frozenset(
        {"ai_runtime_sessions", "ai_runtime_operation_counters", "ai_runtime_operations"}
    ),
    RuntimeDomain.EXECUTION: frozenset(
        {
            "ai_runtime_executions",
            "ai_runtime_idempotency",
            "ai_runtime_events",
            "ai_runtime_operation_counters",
            "ai_runtime_operations",
        }
    ),
    RuntimeDomain.MEMORY: frozenset({"ai_runtime_memories", "ai_runtime_operation_counters", "ai_runtime_operations"}),
    RuntimeDomain.ARTIFACT: frozenset({"ai_runtime_artifacts", "ai_runtime_operation_counters", "ai_runtime_operations"}),
    RuntimeDomain.TASK: frozenset(
        {"ai_runtime_task_graphs", "ai_runtime_task_nodes", "ai_runtime_operation_counters", "ai_runtime_operations"}
    ),
    RuntimeDomain.EVALUATION: frozenset(
        {"ai_runtime_evaluations", "ai_runtime_idempotency", "ai_runtime_operation_counters", "ai_runtime_operations"}
    ),
    RuntimeDomain.RECOVERY: frozenset(
        {
            "ai_runtime_approvals",
            "ai_runtime_external_calls",
            "ai_runtime_recovery_checkpoints",
            "ai_runtime_tool_operations",
            "ai_runtime_operation_counters",
            "ai_runtime_operations",
        }
    ),
}


__all__ = ["build_runtime_sql_metadata", "build_step_sql_metadata", "required_runtime_sql_tables"]
