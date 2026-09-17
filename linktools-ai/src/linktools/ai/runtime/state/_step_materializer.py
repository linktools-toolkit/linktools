#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Materialize Runtime step archives from state routes."""

from collections.abc import Mapping

from ._contracts import ExecutionRepository
from ._model_interaction_store import (
    ModelInteractionInMemoryStepArchive,
    ModelInteractionRuntimeStepStore,
    ModelInteractionStagingStepStore,
    ModelInteractionStateStepArchive,
)
from ._object_router import _RuntimeObjectRouter
from ._plan import RuntimeDomain, RuntimeRetentionMode, RuntimeStatePlan
from ._steps import RuntimeStepStore, StateStepArchive

_STEP_DOMAINS = (
    RuntimeDomain.CONVERSATION,
    RuntimeDomain.EXECUTION,
    RuntimeDomain.RECOVERY,
)


def build_runtime_steps(
    plan: RuntimeStatePlan,
    stores: Mapping[RuntimeDomain, object],
    objects: _RuntimeObjectRouter,
    history_repository: object,
    execution_repository: ExecutionRepository,
    *,
    namespace: str,
    tenant_id: str,
) -> RuntimeStepStore:
    archives: dict[RuntimeDomain, object] = {}
    for domain in _STEP_DOMAINS:
        route = plan.route(domain)
        if (
            route.retention is RuntimeRetentionMode.TRANSIENT
            and domain is not RuntimeDomain.CONVERSATION
        ):
            continue
        if route.retention is RuntimeRetentionMode.DURABLE:
            context_sources = None
            conversation_archive = archives.get(RuntimeDomain.CONVERSATION)
            if isinstance(conversation_archive, StateStepArchive):
                context_sources = {
                    RuntimeDomain.CONVERSATION: conversation_archive.transcript_repository,
                }
            archives[domain] = ModelInteractionStateStepArchive(
                stores[domain],
                object_store=objects.object_store(domain),
                namespace=namespace,
                tenant_id=tenant_id,
                runtime_domain=domain,
                context_sources=context_sources,
                history_repository=(
                    history_repository if domain is RuntimeDomain.CONVERSATION else None
                ),
                execution_repository=(
                    execution_repository if domain is RuntimeDomain.EXECUTION else None
                ),
            )
        else:
            archives[domain] = ModelInteractionInMemoryStepArchive(domain)
    return ModelInteractionRuntimeStepStore(
        ModelInteractionStagingStepStore(),
        conversation_archive=archives[RuntimeDomain.CONVERSATION],
        execution_archive=archives.get(RuntimeDomain.EXECUTION),
        recovery_archive=archives.get(RuntimeDomain.RECOVERY),
        conversation_retention=plan.route(RuntimeDomain.CONVERSATION).retention,
        execution_retention=plan.route(RuntimeDomain.EXECUTION).retention,
        recovery_retention=plan.route(RuntimeDomain.RECOVERY).retention,
    )


__all__ = ["build_runtime_steps"]
