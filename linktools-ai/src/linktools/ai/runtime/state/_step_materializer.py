#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize Runtime step archives from state routes."""

from ...storage import StorageRetentionMode

from collections.abc import Mapping
from functools import partial

from ._contracts import ConversationHistoryRepository, ExecutionRepository
from ._model_interaction_store import (
    ModelInteractionStateStepArchive,
)
from ._object_router import _RuntimeObjectRouter
from ._plan import RuntimeDomain, RuntimeStoragePlan
from ._step_archive import StagingAgentRunStore, StateStepArchive
from ._step_contracts import AgentRunStore
from ._steps import RuntimeAgentRunStore
from ._store import StateStore

_STEP_DOMAINS = (
    RuntimeDomain.CONVERSATION,
    RuntimeDomain.TRACE,
    RuntimeDomain.CHECKPOINT,
)


def build_runtime_agent_run_store(
    plan: RuntimeStoragePlan,
    stores: Mapping[RuntimeDomain, StateStore],
    objects: _RuntimeObjectRouter,
    history_repository: ConversationHistoryRepository,
    execution_repository: ExecutionRepository,
    *,
    namespace: str,
    tenant_id: str,
) -> RuntimeAgentRunStore:
    archives: dict[RuntimeDomain, AgentRunStore] = {}
    for domain in _STEP_DOMAINS:
        route = plan.route(domain)
        if (
            route.retention is StorageRetentionMode.TRANSIENT
            and domain is not RuntimeDomain.CONVERSATION
        ):
            continue
        context_sources = None
        conversation_archive = archives.get(RuntimeDomain.CONVERSATION)
        if isinstance(conversation_archive, StateStepArchive):
            context_sources = {
                RuntimeDomain.CONVERSATION: conversation_archive.transcript_repository,
            }
        archives[domain] = ModelInteractionStateStepArchive(
            stores[domain],
            object_store=objects.object_store(domain),
            archive_object_store=objects.archive_object_store(domain),
            archive_object_reader=partial(objects.read_archive_object, domain),
            object_reader=objects.read_object,
            namespace=namespace,
            tenant_id=tenant_id,
            runtime_domain=domain,
            context_sources=context_sources,
            history_repository=(
                history_repository if domain is RuntimeDomain.CONVERSATION else None
            ),
            execution_repository=(
                execution_repository if domain is RuntimeDomain.TRACE else None
            ),
        )
    return RuntimeAgentRunStore(
        StagingAgentRunStore(),
        execution_repository=execution_repository,
        tenant_id=tenant_id,
        conversation_archive=archives[RuntimeDomain.CONVERSATION],
        execution_archive=archives.get(RuntimeDomain.TRACE),
        recovery_archive=archives.get(RuntimeDomain.CHECKPOINT),
    )


__all__ = ["build_runtime_agent_run_store"]
