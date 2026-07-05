#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""StorageCapabilities: what a given Storage instance can and cannot promise
(spec docs/linktools-ai.md section 10.5). Callers branch on these flags instead
of on concrete Storage/backend types."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StorageCapabilities:
    cross_store_transactions: bool
    optimistic_concurrency: bool
    append_only_events: bool
    distributed_coordination: bool
    full_text_search: bool
    semantic_search: bool
    multi_process_swarm: bool


FILE_STORAGE_CAPABILITIES = StorageCapabilities(
    cross_store_transactions=False,
    optimistic_concurrency=True,
    append_only_events=True,
    distributed_coordination=False,
    full_text_search=False,
    semantic_search=False,
    multi_process_swarm=False,
)

SQLALCHEMY_STORAGE_CAPABILITIES = StorageCapabilities(
    cross_store_transactions=True,
    optimistic_concurrency=True,
    append_only_events=True,
    distributed_coordination=True,
    full_text_search=True,
    semantic_search=False,
    multi_process_swarm=True,
)
