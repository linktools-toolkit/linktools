#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External adapters implementing lower-level ports."""

from ._history import StepExecutionHistoryReader
from ._identity import StaticPrincipalProvider
from ._memory import RuntimeMemoryStore
from ._nats import NatsPublisher
from ._persistence import (
    FilesystemRuntime,
    InMemoryRuntime,
    build_filesystem_runtime,
    build_in_memory_runtime,
)
from ._provider import ProviderClient
from ._runtime_factory import (
    RuntimePersistence,
    open_runtime_persistence,
    runtime_durable_domains,
    runtime_storage_engine,
    runtime_storage_kind,
    runtime_storage_path,
)
from ._schema import (
    build_runtime_sql_metadata,
    build_step_sql_metadata,
    required_runtime_sql_tables,
)
from ._step import (
    FilesystemStepArchive,
    InMemoryStepArchive,
    ObjectMediaAdapter,
    RuntimeStepPersistence,
    SqlStepArchive,
    StagingStepStore,
)

__all__ = [
    "FilesystemStepArchive", "FilesystemRuntime", "InMemoryRuntime", "InMemoryStepArchive",
    "NatsPublisher", "ObjectMediaAdapter", "ProviderClient", "RuntimeMemoryStore", "RuntimeStepPersistence", "SqlStepArchive",
    "RuntimePersistence", "StaticPrincipalProvider", "StepExecutionHistoryReader", "StagingStepStore", "build_filesystem_runtime", "build_in_memory_runtime", "build_runtime_sql_metadata", "build_step_sql_metadata", "open_runtime_persistence", "required_runtime_sql_tables", "runtime_durable_domains", "runtime_storage_engine", "runtime_storage_kind", "runtime_storage_path",
]
