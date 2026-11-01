#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared local composition for AI CLI commands."""

import asyncio
import os
from argparse import Namespace
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import CommandError
from linktools.cli.argparse import ConfigAction
from linktools.core import ConfigField, environ

from linktools.ai.asset import (
    AssetStore,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.core import DEFAULT_DISCOVERY_POLICY
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_metrics_sqlite
from linktools.ai.observe import Metrics
from linktools.ai.storage import FilesystemMutationLock, StorageOverlay
from linktools.ai.workspace import Workspace

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
    from linktools.cli import CommandParser
    from linktools.ai.model import ModelBinding
    from linktools.ai.runtime import Runtime
    from linktools.ai.persistence import PersistenceStorage

MODEL_APIS = ("openai", "openai-responses", "anthropic")
MODEL_API = ConfigField(name="MODEL_API", cast=str, default=None)
_MODEL_NUMERIC_OPTIONS = (
    ("context_window", int, "context window in tokens"),
    ("max_tokens", int, "maximum output tokens"),
    ("timeout", float, "provider HTTP timeout in seconds (not a run deadline)"),
    ("max_retries", int, "maximum additional transport retries"),
    ("retry_delay", float, "initial retry delay in seconds"),
    ("retry_max_delay", float, "local retry delay cap in seconds"),
    ("retry_backoff_factor", float, "retry delay multiplier (at least 1)"),
)


def add_local_runtime_arguments(parser: "CommandParser") -> None:
    parser.add_argument("--project", type=Path, default=None, help="working directory")
    parser.add_argument("--model-api", action=ConfigAction, config=MODEL_API, choices=MODEL_APIS,
                        help="model API protocol (default: openai)")
    parser.add_argument("--base-url", default=None, help="selected protocol endpoint")
    parser.add_argument("--model", default=None, help="selected protocol model name")
    parser.add_argument("--api-key", default=None, help="one-shot selected protocol credential")
    for name, value_type, description in _MODEL_NUMERIC_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=value_type, default=None, help=description)


def _load_workspace(root: "Path | None" = None) -> Workspace:
    start = Path.cwd()
    try:
        return Workspace.discover(start) if root is None else Workspace.discover(start, root=root)
    except AIError as error:
        if error.code is not ErrorCode.WORKSPACE_CONFIG_INVALID:
            raise
        return Workspace.initialize(start if root is None else root)


def _local_runtime_root(workspace: Workspace) -> Path:
    return workspace.storage_root / "runtime"


def _local_runtime_assets(workspace: Workspace) -> AssetStore:
    return AssetStore(
        StorageOverlay(
            DirectoryAssetBackend(
                str(workspace.storage_root),
                path_adapter=PrefixAssetPathAdapter(
                    {"agent": "agents", "skill": "skills", "mcp": "mcp"}
                ),
                kinds=("agent", "skill", "mcp"),
                follow_external_symlinks=True,
                ignore_paths=DEFAULT_DISCOVERY_POLICY.ignores,
            )
        )
    )


def _local_runtime_storage(workspace: Workspace) -> "PersistenceStorage":
    from linktools.ai.persistence import PersistenceStorage

    return PersistenceStorage.from_root(_local_runtime_root(workspace))


async def _local_metrics(workspace: Workspace) -> Metrics:
    runtime_root = _local_runtime_root(workspace)
    runtime_root.mkdir(parents=True, exist_ok=True)
    path = runtime_root / "metrics.db"
    async with FilesystemMutationLock(str(path) + ".lock"):
        if not path.exists():
            await provision_metrics_sqlite(path)
    return Metrics.sqlite(path, namespace="default")


def _local_input_config(
    workspace: Workspace,
) -> tuple[str, ...]:

    try:
        values = workspace.config.get("input_types", [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError("input_types must be a list of input type strings")
        return tuple(values)
    except (TypeError, ValueError) as error:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            "Invalid workspace input_types configuration",
            retryable=False,
            safe_details={"reason": "model_input_configuration_invalid"},
        ) from error


@dataclass(frozen=True, slots=True)
class _ModelOptions:
    api: str
    model: str
    base_url: str | None
    api_key: str | None = field(repr=False)
    input_types: tuple[str, ...]
    limits: dict[str, int | float]

    @classmethod
    def load(cls, workspace: Workspace, args: Namespace | None = None) -> "_ModelOptions":
        arguments = {} if args is None else vars(args)
        api = (arguments.get("model_api") or environ.config.get(MODEL_API)
               or workspace.config.get("model_api") or os.getenv("MODEL_API") or "openai")
        if api not in MODEL_APIS:
            raise AIError(ErrorCode.MODEL_CONFIG_INVALID, "model_api must be one of: " + ", ".join(MODEL_APIS))
        prefix = "ANTHROPIC" if api == "anthropic" else "OPENAI"
        options: dict[str, str | None] = {}
        for name in ("model", "base_url", "api_key"):
            config_field = ConfigField(name=f"{prefix}_{name.upper()}", cast=str, default=None, secret=name == "api_key")
            value = arguments.get(name)
            if value is None:
                value = environ.config.get(config_field)
            if name == "model" and (not isinstance(value, str) or not value.strip()):
                value = workspace.config.get("model")
            if value is None:
                value = os.getenv(config_field.name, "").strip() or None
            if name != "model" and (value is None or not value.strip()):
                value = os.getenv(config_field.name) or None
            options[name] = value
        model = options["model"]
        model = model.strip() if isinstance(model, str) and model.strip() else os.getenv(f"{prefix}_MODEL", "").strip()
        input_types = _local_input_config(workspace)
        limits: dict[str, int | float] = {}
        try:
            for name, value_type, _description in _MODEL_NUMERIC_OPTIONS:
                value = arguments.get(name)
                if value is None:
                    value = environ.config.get(ConfigField(name="MODEL_" + name.upper(), cast=None, default=None))
                    if isinstance(value, str):
                        value = value_type(value)
                    if value is None:
                        value = workspace.config.get(name)
                if value is not None:
                    limits[name] = value
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.MODEL_CONFIG_INVALID, "Invalid local model limits",
                          safe_details={"reason": "model_configuration_invalid"}) from error
        return cls(api, model, options["base_url"], options["api_key"], input_types, limits)

    def create_model(self) -> "ModelBinding":
        from linktools.ai.model import create_model

        if not self.model:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "model is required")
        try:
            return create_model(
                api=self.api, model=self.model, input_types=self.input_types,
                base_url=self.base_url, api_key=self.api_key,
                model_settings={name: value for name, value in self.limits.items() if name in {"max_tokens", "timeout"}},
                **{name: value for name, value in self.limits.items() if name not in {"max_tokens", "timeout"}},
            )
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.MODEL_CONFIG_INVALID, "Invalid local model configuration",
                          safe_details={"reason": "model_configuration_invalid"}) from error

    def status(self, model: "ModelBinding") -> dict[str, object]:
        return {
            "model": model.model_identity.removeprefix(model.provider + ":"),
            "model_api": self.api,
            "input_mime_types": model.contract.get("input_mime_types"),
            "base_url_configured": bool(self.base_url),
            "api_key_configured": bool(self.api_key),
        }


@dataclass(frozen=True, slots=True)
class LocalRuntime:
    """Borrowed resources and metadata valid for one command invocation."""

    runtime: "Runtime"
    metrics: Metrics
    capabilities: tuple[dict[str, str | int], ...]
    status: "Mapping[str, object]"


@asynccontextmanager
async def _open_local_runtime(
    workspace: Workspace,
    *,
    model: "ModelBinding",
    status: "Mapping[str, object] | None" = None,
) -> "AsyncIterator[LocalRuntime]":
    from linktools.ai.capability import AssetCapability, FileSystem, MemoryCapability, ShellCapability
    from linktools.ai.runtime import RuntimeFeatures
    from linktools.ai.capability import CheckpointCapability, ExecutionCapability, TraceCapability
    from linktools.ai.runtime import Runtime

    async with AssetCapability.open(_local_runtime_assets(workspace)) as assets:
        features = RuntimeFeatures("workspace")
        features.model("default", model)
        features.capability(assets)
        features.capability(FileSystem(workspace))
        features.capability(ShellCapability())
        features.capability(MemoryCapability())
        features.capability(ExecutionCapability(retain_history=True))
        features.capability(CheckpointCapability())
        features.capability(TraceCapability())
        metrics = await _local_metrics(workspace)
        capture = await features.capture()
        declarations: tuple[dict[str, str | int], ...] = tuple(
            {"kind": item.kind, "id": item.id, "revision": item.revision} for item in capture.contributions
        )
        if not any(item["kind"] == "agent" and item["id"] == "default" for item in declarations):
            declarations = ({"kind": "agent", "id": "default", "revision": 1}, *declarations)
        async with Runtime.open(
            "default", storage=_local_runtime_storage(workspace),
            features=capture,
            metrics=metrics,
        ) as runtime:
            yield LocalRuntime(runtime, metrics, declarations, {
                "workspace": str(workspace.root), "asset_root": str(workspace.storage_root),
                **({} if status is None else status),
            })


def run_local(args: Namespace, execute: "Callable[[LocalRuntime], Awaitable[int]]") -> int:
    """Run a command while owning its Runtime and local resource lifetime."""
    async def run() -> int:
        workspace = _load_workspace(args.project)
        options = _ModelOptions.load(workspace, args)
        model = options.create_model()
        async with _open_local_runtime(workspace, model=model, status=options.status(model)) as local:
            return await execute(local)

    try:
        return asyncio.run(run())
    except (AIError, ImportError, TypeError, ValueError) as error:
        raise CommandError(str(error)) from error


__all__ = ["LocalRuntime", "add_local_runtime_arguments", "run_local"]
