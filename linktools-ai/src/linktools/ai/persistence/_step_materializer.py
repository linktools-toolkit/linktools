#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize Runtime step archives from state routes."""

from ..storage import StorageRetentionMode

from collections.abc import Mapping
from functools import partial

from .contracts import ConversationHistoryRepository, ExecutionRepository, ToolStateRepository
from ._checkpoint_repositories import ToolRepositoryImpl
from ._model_interaction_store import (
    ModelInteractionStateStepArchive,
)
from ._object_router import _RuntimeObjectRouter
from ._plan import PersistenceDomain, PersistencePlan
from ._step_archive import StagingAgentRunStore, StateStepArchive
from ._step_contracts import AgentRunStore
from ._sql import SqlStateStore
from ._step_store import SqlStepStore
from ._run_registration import RunRegistrationStore
from .runs import RuntimeAgentRunStore
from .state import StateStore

_STEP_DOMAINS = (
    PersistenceDomain.CONVERSATION,
    PersistenceDomain.TRACE,
)


def build_runtime_agent_run_store(
    plan: PersistencePlan,
    stores: Mapping[PersistenceDomain, StateStore],
    objects: _RuntimeObjectRouter,
    history_repository: ConversationHistoryRepository,
    execution_repository: ExecutionRepository,
    *,
    namespace: str,
    tenant_id: str,
    run_registrations: RunRegistrationStore | None = None,
    tool_repository: ToolStateRepository | None = None,
) -> RuntimeAgentRunStore:
    if run_registrations is None:
        candidates = (PersistenceDomain.EXECUTION, PersistenceDomain.TRACE, PersistenceDomain.CONVERSATION)
        domain = next((domain for domain in candidates
                       if plan.route(domain).retention is StorageRetentionMode.DURABLE), PersistenceDomain.EXECUTION)
        owner = stores[domain]
        native = (SqlStepStore(owner.context, namespace=namespace, tenant_id=tenant_id)
                  if isinstance(owner, SqlStateStore) else None)
        run_registrations = RunRegistrationStore(owner, namespace=namespace, tenant_id=tenant_id, native=native)
    archives: dict[PersistenceDomain, AgentRunStore] = {}
    for domain in _STEP_DOMAINS:
        route = plan.route(domain)
        if (
            route.retention is StorageRetentionMode.TRANSIENT
            and domain is not PersistenceDomain.CONVERSATION
        ):
            continue
        context_sources = None
        conversation_archive = archives.get(PersistenceDomain.CONVERSATION)
        if isinstance(conversation_archive, StateStepArchive):
            context_sources = {
                PersistenceDomain.CONVERSATION: conversation_archive.transcript_repository,
            }
        archives[domain] = ModelInteractionStateStepArchive(
            stores[domain],
            object_store=objects.object_store(domain),
            cache=route.cache,
            archive_object_store=objects.archive_object_store(domain),
            archive_object_reader=partial(objects.read_archive_object, domain),
            object_reader=objects.read_object,
            namespace=namespace,
            tenant_id=tenant_id,
            runtime_domain=domain,
            run_registrations=run_registrations,
            context_sources=context_sources,
            history_repository=(
                history_repository if domain is PersistenceDomain.CONVERSATION else None
            ),
            execution_repository=(
                execution_repository if domain is PersistenceDomain.TRACE else None
            ),
        )
    trace_archive = archives.get(PersistenceDomain.TRACE)
    if isinstance(trace_archive, StateStepArchive) and type(tool_repository) is ToolRepositoryImpl:
        trace_archive.bind_tool_repository(tool_repository)
    native_by_engine: dict[object, SqlStepStore] = {}
    for domain, archive in archives.items():
        state = stores[domain]
        if isinstance(state, SqlStateStore):
            native = native_by_engine.get(state.context.engine)
            if native is None:
                native = SqlStepStore(state.context, namespace=namespace, tenant_id=tenant_id)
                native_by_engine[state.context.engine] = native
            archive.bind_native_run_store(native)
    trace = archives.get(PersistenceDomain.TRACE)
    execution = stores[PersistenceDomain.EXECUTION]
    if (isinstance(trace, StateStepArchive) and isinstance(trace.state_store, SqlStateStore)
            and isinstance(execution, SqlStateStore)
            and trace.state_store.context.engine is execution.context.engine):
        trace.enable_native_run_reads()
    return RuntimeAgentRunStore(
        StagingAgentRunStore(),
        execution_repository=execution_repository,
        tenant_id=tenant_id,
        run_registrations=run_registrations,
        conversation_archive=archives[PersistenceDomain.CONVERSATION],
        execution_archive=archives.get(PersistenceDomain.TRACE),
    )


__all__ = ["build_runtime_agent_run_store"]
