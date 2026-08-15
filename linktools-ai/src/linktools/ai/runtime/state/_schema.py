#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned SQL metadata and plan-to-table selection."""

from typing import TYPE_CHECKING

from ...storage import (
    build_object_sql_metadata,
    sql_digest,
    sql_integer_id,
    sql_table_options,
    sql_text_key,
)
from ._plan import (
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeStatePlan,
    RuntimeStateRoute,
)

if TYPE_CHECKING:
    from sqlalchemy import MetaData

    from ...storage import ObjectStore


_STEP_DOMAINS = frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY})
_OBJECT_DOMAINS = frozenset({RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.MEMORY, RuntimeDomain.ARTIFACT, RuntimeDomain.RECOVERY})


def required_runtime_sql_tables(plan: RuntimeStatePlan, *, include_object_tables: bool | None = None) -> frozenset[str]:
    names: set[str] = set()
    for domain in RuntimeDomain:
        if plan.route(domain).retention is not RuntimeRetentionMode.DURABLE:
            continue
        names.update(_DOMAIN_TABLES[domain])
    if include_object_tables is None:
        include_object_tables = any(
            plan.route(domain).retention is RuntimeRetentionMode.DURABLE
            for domain in _OBJECT_DOMAINS
        )
    if include_object_tables and any(
        plan.route(domain).retention is RuntimeRetentionMode.DURABLE
        for domain in _OBJECT_DOMAINS
    ):
        names.update({"storage_objects", "storage_object_chunks"})
    return frozenset(names)


def build_runtime_sql_metadata(plan: "RuntimeStatePlan | frozenset[RuntimeDomain]", *, include_object_tables: bool | None = None) -> "MetaData":
    from sqlalchemy import (
        JSON,
        Boolean,
        Column,
        DateTime,
        Index,
        MetaData,
        Table,
        Text,
        UniqueConstraint,
        func,
    )

    if isinstance(plan, frozenset):
        routes = {domain.value: RuntimeStateRoute.memory() for domain in RuntimeDomain}
        for domain in plan:
            routes[domain.value] = RuntimeStateRoute.filesystem(f"runtime-{domain.value}")
        plan = RuntimeStatePlan(**routes)
    metadata = MetaData()
    tables = required_runtime_sql_tables(plan, include_object_tables=include_object_tables)
    common = {
        "namespace_key": sql_digest(),
        "tenant_id": sql_text_key(256),
    }

    def table(
        name: str,
        fields: dict[str, object],
        unique: tuple[str, ...],
        *,
        nullable: frozenset[str] = frozenset(),
        updated: bool = True,
        extra_constraints: tuple[object, ...] = (),
    ) -> None:
        columns = [Column("id", sql_integer_id(), primary_key=True, autoincrement=True)]
        for field, value in fields.items():
            columns.append(Column(field, value, nullable=field in nullable))
        columns.append(Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()))
        if updated:
            columns.append(Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()))
        constraints = [UniqueConstraint(*unique, name=f"uk_{name}_{'_'.join(unique)}"), *extra_constraints]
        Table(name, metadata, *columns, *constraints, **sql_table_options())

    if "runtime_sessions" in tables:
        table("runtime_sessions", {**common, "session_id": sql_text_key(), "owner_principal_id": sql_text_key(), "binding_digest": sql_digest(), "status": sql_text_key(64), "revision": sql_integer_id(), "resource_generation": sql_integer_id(), "cwd": Text(), "metadata_json": JSON(), "continuation_step_run_id": sql_text_key(), "closed_at": DateTime(timezone=True)}, ("namespace_key", "tenant_id", "session_id"), nullable=frozenset({"cwd", "continuation_step_run_id", "closed_at"}))
    if "runtime_executions" in tables:
        table("runtime_executions", {**common, "execution_id": sql_text_key(), "session_id": sql_text_key(), "binding_digest": sql_digest(), "parent_execution_id": sql_text_key(), "root_execution_id": sql_text_key(), "source_execution_id": sql_text_key(), "base_execution_id": sql_text_key(), "lineage_kind": sql_text_key(64), "status": sql_text_key(64), "revision": sql_integer_id(), "event_sequence": sql_integer_id(), "agent_run_sequence": sql_integer_id(), "error_code": sql_text_key(128), "safe_error_details_json": JSON(), "memory_scope": sql_text_key(), "conversation_step_run_id": sql_text_key(), "output_schema_id": sql_text_key(), "output_schema_revision": sql_integer_id(), "output_schema_fingerprint": sql_digest(), "result_store_id": sql_text_key(128), "result_object_key": sql_text_key(1024), "result_digest": sql_digest(), "result_size": sql_integer_id(), "stop_reason": Text(), "input_tokens": sql_integer_id(), "output_tokens": sql_integer_id(), "total_cost_micros": sql_integer_id(), "result_created_at": DateTime(timezone=True)}, ("namespace_key", "tenant_id", "execution_id"), nullable=frozenset({"session_id", "parent_execution_id", "source_execution_id", "base_execution_id", "error_code", "memory_scope", "conversation_step_run_id", "output_schema_id", "output_schema_revision", "output_schema_fingerprint", "result_store_id", "result_object_key", "result_digest", "result_size", "stop_reason", "input_tokens", "output_tokens", "total_cost_micros", "result_created_at"}))
    if "runtime_idempotency" in tables:
        table("runtime_idempotency", {**common, "runtime_domain": sql_text_key(64), "scope": sql_text_key(128), "key_hash": sql_digest(), "request_digest": sql_digest(), "resource_kind": sql_text_key(128), "resource_id": sql_text_key(), "status": sql_text_key(64), "result_digest": sql_digest(), "error_code": sql_text_key(128)}, ("namespace_key", "tenant_id", "runtime_domain", "scope", "key_hash"), nullable=frozenset({"result_digest", "error_code"}))
    if "runtime_events" in tables:
        table("runtime_events", {**common, "execution_id": sql_text_key(), "sequence": sql_integer_id(), "event_type": sql_text_key(128), "payload_json": JSON()}, ("namespace_key", "tenant_id", "execution_id", "sequence"), updated=False)
    if "runtime_task_graphs" in tables:
        table("runtime_task_graphs", {**common, "graph_id": sql_text_key(), "status": sql_text_key(64), "revision": sql_integer_id()}, ("namespace_key", "tenant_id", "graph_id"))
    if "runtime_task_nodes" in tables:
        table(
            "runtime_task_nodes",
            {
                **common,
                "graph_id": sql_text_key(),
                "node_id": sql_text_key(),
                "dependencies_json": JSON(),
                "input_json": JSON(),
                "budget_cost": sql_integer_id(),
                "status": sql_text_key(64),
                "revision": sql_integer_id(),
                "owner": sql_text_key(),
                "fence": sql_integer_id(),
                "lease_expires_at": DateTime(timezone=True),
                "execution_id": sql_text_key(),
                "result_digest": sql_digest(),
                "error_code": sql_text_key(128),
                "error_digest": sql_digest(),
            },
            ("namespace_key", "tenant_id", "graph_id", "node_id"),
            nullable=frozenset(
                {
                    "input_json",
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
    if "runtime_evaluations" in tables:
        table("runtime_evaluations", {**common, "evaluation_id": sql_text_key(), "execution_id": sql_text_key(), "dataset_id": sql_text_key(), "dataset_revision": sql_integer_id(), "evaluator_id": sql_text_key(), "evaluator_revision": sql_integer_id(), "binding_digest": sql_digest(), "output_schema_fingerprint": sql_digest(), "artifact_digest": sql_digest(), "status": sql_text_key(64), "revision": sql_integer_id(), "metrics_json": JSON()}, ("namespace_key", "tenant_id", "evaluation_id"), nullable=frozenset({"artifact_digest"}))
    if "runtime_memories" in tables:
        table("runtime_memories", {**common, "memory_id": sql_text_key(), "memory_scope_key": sql_digest(), "object_store_id": sql_text_key(128), "object_key": sql_text_key(1024), "object_digest": sql_digest(), "object_size": sql_integer_id(), "metadata_json": JSON(), "revision": sql_integer_id()}, ("namespace_key", "tenant_id", "memory_id"))
    if "runtime_artifacts" in tables:
        table("runtime_artifacts", {**common, "artifact_id": sql_text_key(), "execution_id": sql_text_key(), "producer": sql_text_key(), "media_type": sql_text_key(), "object_store_id": sql_text_key(128), "object_key": sql_text_key(1024), "object_digest": sql_digest(), "object_size": sql_integer_id()}, ("namespace_key", "tenant_id", "artifact_id"))
    if "runtime_approvals" in tables:
        table("runtime_approvals", {**common, "approval_id": sql_text_key(), "execution_id": sql_text_key(), "operation_id": sql_text_key(), "status": sql_text_key(64), "idempotency_key_hash": sql_digest(), "decision": sql_text_key(64), "decided_by": sql_text_key(), "decision_digest": sql_digest(), "decided_at": DateTime(timezone=True)}, ("namespace_key", "tenant_id", "approval_id"), nullable=frozenset({"idempotency_key_hash", "decision", "decided_by", "decision_digest", "decided_at"}), updated=False)
    if "runtime_external_calls" in tables:
        table("runtime_external_calls", {**common, "call_id": sql_text_key(), "execution_id": sql_text_key(), "operation_id": sql_text_key(), "status": sql_text_key(64), "idempotency_key_hash": sql_digest(), "object_store_id": sql_text_key(128), "object_key": sql_text_key(1024), "object_digest": sql_digest(), "object_size": sql_integer_id(), "payload_digest": sql_digest(), "supplied_at": DateTime(timezone=True)}, ("namespace_key", "tenant_id", "call_id"), nullable=frozenset({"idempotency_key_hash", "object_store_id", "object_key", "object_digest", "object_size", "payload_digest", "supplied_at"}), updated=False)
    if "runtime_recovery_checkpoints" in tables:
        table("runtime_recovery_checkpoints", {**common, "execution_id": sql_text_key(), "step_run_id": sql_text_key(), "agent_run_sequence": sql_integer_id(), "state": sql_text_key(64), "handoff_phase": sql_text_key(64), "input_json": JSON(), "terminal_handoff_json": JSON(), "handoff_contract_digest": sql_digest(), "pending_operation_id": sql_text_key(), "revision": sql_integer_id()}, ("namespace_key", "tenant_id", "execution_id"), nullable=frozenset({"terminal_handoff_json", "handoff_contract_digest", "pending_operation_id"}))
    if "runtime_tool_operations" in tables:
        table("runtime_tool_operations", {**common, "tool_operation_id": sql_text_key(), "step_run_id": sql_text_key(), "tool_call_id": sql_text_key(), "idempotency_key_hash": sql_digest(), "tool_name": sql_text_key(), "arguments_hash": sql_digest(), "binding_fingerprint": sql_digest(), "replay_safe": Boolean(), "status": sql_text_key(64), "owner": sql_text_key(), "fence": sql_integer_id(), "lease_expires_at": DateTime(timezone=True), "result_store_id": sql_text_key(128), "result_object_key": sql_text_key(1024), "result_digest": sql_digest(), "result_size": sql_integer_id(), "error_code": sql_text_key(128)}, ("namespace_key", "tenant_id", "tool_operation_id"), nullable=frozenset({"owner", "lease_expires_at", "result_store_id", "result_object_key", "result_digest", "result_size", "error_code"}), extra_constraints=(UniqueConstraint("namespace_key", "tenant_id", "step_run_id", "tool_call_id", name="uk_runtime_tool_operations_step_call"),))
    if "runtime_operation_counters" in tables:
        table("runtime_operation_counters", {**common, "runtime_domain": sql_text_key(64), "resource_kind": sql_text_key(128), "resource_id": sql_text_key(), "last_sequence": sql_integer_id()}, ("namespace_key", "tenant_id", "runtime_domain", "resource_kind", "resource_id"))
    if "runtime_operations" in tables:
        table("runtime_operations", {**common, "runtime_domain": sql_text_key(64), "resource_kind": sql_text_key(128), "resource_id": sql_text_key(), "operation_kind": sql_text_key(128), "operation_id": sql_text_key(), "sequence": sql_integer_id(), "status": sql_text_key(64), "execution_id": sql_text_key(), "request_digest": sql_digest(), "result_ref": Text(), "result_digest": sql_digest(), "error_code": sql_text_key(128), "compactable": Boolean()}, ("namespace_key", "tenant_id", "runtime_domain", "operation_id"), nullable=frozenset({"execution_id", "result_ref", "result_digest", "error_code"}), extra_constraints=(UniqueConstraint("namespace_key", "tenant_id", "runtime_domain", "resource_kind", "resource_id", "sequence", name="uk_runtime_operations_stream"),))
    if "step_runs" in tables:
        table("step_runs", {**common, "runtime_domain": sql_text_key(64), "step_run_id": sql_text_key(), "harness_conversation_id": sql_text_key(), "parent_step_run_id": sql_text_key(), "agent_name": sql_text_key(), "metadata_json": JSON(), "last_event_index": sql_integer_id(), "last_snapshot_index": sql_integer_id(), "last_effect_index": sql_integer_id(), "started_at": DateTime(timezone=True)}, ("namespace_key", "tenant_id", "runtime_domain", "step_run_id"), nullable=frozenset({"harness_conversation_id", "parent_step_run_id", "agent_name"}))
    if "step_events" in tables:
        table("step_events", {**common, "step_run_id": sql_text_key(), "event_index": sql_integer_id(), "event_kind": sql_text_key(), "step_index": sql_integer_id(), "timestamp": DateTime(timezone=True), "harness_conversation_id": sql_text_key(), "parent_step_run_id": sql_text_key(), "agent_name": sql_text_key(), "tool_call_id": sql_text_key(), "tool_name": sql_text_key(), "error": Text(), "metadata_json": JSON()}, ("namespace_key", "tenant_id", "step_run_id", "event_index"), nullable=frozenset({"harness_conversation_id", "parent_step_run_id", "agent_name", "tool_call_id", "tool_name", "error"}), updated=False)
    if "step_snapshots" in tables:
        table("step_snapshots", {**common, "runtime_domain": sql_text_key(64), "step_run_id": sql_text_key(), "snapshot_index": sql_integer_id(), "step_index": sql_integer_id(), "state": sql_text_key(64), "harness_conversation_id": sql_text_key(), "parent_step_run_id": sql_text_key(), "agent_name": sql_text_key(), "timestamp": DateTime(timezone=True), "object_store_id": sql_text_key(128), "messages_json": JSON()}, ("namespace_key", "tenant_id", "runtime_domain", "step_run_id", "snapshot_index"), nullable=frozenset({"harness_conversation_id", "parent_step_run_id", "agent_name"}))
    if "step_effects" in tables:
        table("step_effects", {**common, "step_run_id": sql_text_key(), "effect_index": sql_integer_id(), "tool_call_id": sql_text_key(), "tool_name": sql_text_key(), "status": sql_text_key(64), "started_at": DateTime(timezone=True), "ended_at": DateTime(timezone=True), "idempotency_key": sql_text_key(), "effect_summary": Text()}, ("namespace_key", "tenant_id", "step_run_id", "effect_index"), nullable=frozenset({"ended_at", "idempotency_key", "effect_summary"}))
        Index("ix_step_effects_tool", metadata.tables["step_effects"].c.namespace_key, metadata.tables["step_effects"].c.tenant_id, metadata.tables["step_effects"].c.step_run_id, metadata.tables["step_effects"].c.tool_call_id, metadata.tables["step_effects"].c.effect_index)
    if "storage_objects" in tables:
        object_metadata = build_object_sql_metadata()
        for item in object_metadata.tables.values():
            item.to_metadata(metadata)
    return metadata


def build_step_sql_metadata(runtime_domain: RuntimeDomain, *, object_store: "ObjectStore | None" = None) -> "MetaData":
    if runtime_domain not in _STEP_DOMAINS:
        raise ValueError("Step archive owner is invalid")
    route = RuntimeStateRoute.filesystem("runtime")
    routes = {domain.value: RuntimeStateRoute.memory() for domain in RuntimeDomain}
    routes[runtime_domain.value] = route
    plan = RuntimeStatePlan(**routes)
    full_metadata = build_runtime_sql_metadata(plan)
    from sqlalchemy import MetaData

    metadata = MetaData()
    names = {"step_runs", "step_snapshots"}
    if runtime_domain is RuntimeDomain.EXECUTION:
        names.add("step_events")
    if runtime_domain is RuntimeDomain.RECOVERY:
        names.add("step_effects")
    if object_store is None:
        names.update({"storage_objects", "storage_object_chunks"})
    for name in names:
        full_metadata.tables[name].to_metadata(metadata)
    return metadata


_DOMAIN_TABLES = {
    RuntimeDomain.CONVERSATION: frozenset({"runtime_sessions", "runtime_operation_counters", "runtime_operations", "step_runs", "step_snapshots"}),
    RuntimeDomain.EXECUTION: frozenset({"runtime_executions", "runtime_idempotency", "runtime_events", "runtime_operation_counters", "runtime_operations", "step_runs", "step_events", "step_snapshots"}),
    RuntimeDomain.MEMORY: frozenset({"runtime_memories", "runtime_operation_counters", "runtime_operations"}),
    RuntimeDomain.ARTIFACT: frozenset({"runtime_artifacts", "runtime_operation_counters", "runtime_operations"}),
    RuntimeDomain.TASK: frozenset({"runtime_task_graphs", "runtime_task_nodes", "runtime_operation_counters", "runtime_operations"}),
    RuntimeDomain.EVALUATION: frozenset({"runtime_evaluations", "runtime_idempotency", "runtime_operation_counters", "runtime_operations"}),
    RuntimeDomain.RECOVERY: frozenset({"runtime_approvals", "runtime_external_calls", "runtime_recovery_checkpoints", "runtime_tool_operations", "runtime_operation_counters", "runtime_operations", "step_runs", "step_snapshots", "step_effects"}),
}

__all__ = ["build_runtime_sql_metadata", "build_step_sql_metadata", "required_runtime_sql_tables"]
