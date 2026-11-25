#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared local composition for AI CLI commands."""

import asyncio
import os
from argparse import Namespace
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import CommandError
from linktools.cli.argparse import ConfigAction
from linktools.core import ConfigField, environ
from linktools.errors import ConfigCastError

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
    from linktools.ai.core import JsonValue
    from linktools.ai.runtime import Runtime
    from linktools.ai.session import SessionTurnNotice, ToolProcessItem
    from linktools.ai.persistence import PersistenceStorage

MODEL_APIS = ("openai", "openai-responses", "anthropic")
MODEL_API = ConfigField(name="MODEL_API", cast=str, default=None)
MODEL_VISION = ConfigField(name="MODEL_VISION", cast=bool)
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


def _local_input_config(workspace: Workspace) -> tuple[str, ...]:
    try:
        values = workspace.config.get("input_types", [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError("input_types must be a list of input type strings")
        vision = environ.config.get(MODEL_VISION, default=None)
        if vision is None:
            return tuple(values)
        if vision:
            return (*values, "image")
        return tuple(item for item in values if item != "image" and not (
            item.startswith("mime:") and item[5:].strip().lower().startswith("image/")
        ))
    except (ConfigCastError, TypeError, ValueError) as error:
        raise AIError(
            ErrorCode.MODEL_CONFIG_INVALID,
            "Invalid workspace input_types or LINKTOOLS_MODEL_VISION configuration",
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

    def create_model_binding(self) -> "ModelBinding":
        from linktools.ai.model import create_model_binding

        if not self.model:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "model is required")
        try:
            return create_model_binding(
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
    workspace: Workspace


class _LocalSessionFeed:
    def __init__(self, runtime: "Runtime") -> None:
        self.runtime = runtime
        self._callbacks: dict[str, set["Callable[[str, ToolProcessItem], None]"]] = {}

    def subscribe(
        self, session_id: str, on_process: "Callable[[str, ToolProcessItem], None]",
    ) -> "Callable[[], None]":
        callbacks = self._callbacks.setdefault(session_id, set())
        callbacks.add(on_process)

        def unsubscribe() -> None:
            callbacks.discard(on_process)
            if not callbacks:
                self._callbacks.pop(session_id, None)

        return unsubscribe

    def receive(self, value: dict[str, object]) -> None:
        # This authenticated local carrier is advisory: never install its
        # revision, occupancy, or terminal state as authoritative.
        if value.get("kind") == "ordinary_turn":
            session_id, turn_id, revision = value.get("session_id"), value.get("turn_id"), value.get("revision")
            if (
                self.runtime.sessions.supports_turn_sessions
                and value.get("namespace") == self.runtime.namespace
                and value.get("tenant_id") == self.runtime.tenant_id
                and value.get("phase") in ("admitted", "terminal")
                and isinstance(session_id, str) and isinstance(turn_id, str)
                and type(revision) is int and revision >= 1
            ):
                self.runtime.sessions.invalidate_turn_session(session_id, turn_id, revision=revision)
            return
        kind, session_id, execution_id = value.get("kind"), value.get("session_id"), value.get("execution_id")
        if not isinstance(session_id, str) or not session_id or not isinstance(execution_id, str) or not execution_id:
            return
        if kind in ("admitted", "terminal"):
            self.runtime.sessions.invalidate(session_id)
        elif kind == "tool" and self._callbacks.get(session_id):
            row = value.get("process")
            if not isinstance(row, dict):
                return
            from linktools.ai.session import ToolProcessItem

            item = ToolProcessItem.from_notice(row)
            if item is None:
                return
            for callback in tuple(self._callbacks[session_id]):
                callback(execution_id, item)


@asynccontextmanager
async def local_turn_notices(local: LocalRuntime, *, receive: bool = False) -> "AsyncIterator[_LocalSessionFeed | None]":
    from ._session_notices import session_notice_publisher, session_notice_receiver

    feed = _LocalSessionFeed(local.runtime) if receive else None
    runtime_root = _local_runtime_root(local.workspace)
    async with AsyncExitStack() as stack:
        receiver = None if feed is None else await stack.enter_async_context(session_notice_receiver(runtime_root, feed.receive))
        offer = await stack.enter_async_context(session_notice_publisher(runtime_root, exclude=receiver))

        def publish(notice: "SessionTurnNotice") -> None:
            offer(asdict(notice), discover=notice.kind != "tool")

        stack.callback(local.runtime.sessions.observe_turns(publish))
        if local.runtime.sessions.supports_turn_sessions:
            def publish_lifecycle(notice: "dict[str, JsonValue]") -> None:
                offer({
                    **notice, "namespace": local.runtime.namespace, "tenant_id": local.runtime.tenant_id,
                })

            stack.callback(local.runtime.sessions.observe_turn_lifecycle(publish_lifecycle))
        yield feed


@asynccontextmanager
async def _open_local_runtime(
    workspace: Workspace,
    *,
    model: "ModelBinding",
    status: "Mapping[str, object] | None" = None,
) -> "AsyncIterator[LocalRuntime]":
    from linktools.ai.capability import (
        AssetCapability, ExecutionCapability, FileSystemCapability, MemoryCapability,
        ShellCapability, StepPersistenceCapability, TraceCapability,
    )
    from linktools.ai.core import PrincipalKind, ResourceKind, TenantAuthorizationPolicy
    from linktools.ai.runtime import Runtime, RuntimeFeatures

    class LocalWorkspaceAuthorization(TenantAuthorizationPolicy):
        async def authorize(self, principal, action, resource) -> None:
            if (
                resource.kind is ResourceKind.SESSION
                and resource.owner_principal_id is None
                and principal.kind != PrincipalKind.LOCAL_TRUSTED.value
            ):
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            await super().authorize(principal, action, resource)

    features = RuntimeFeatures("workspace")
    features.model("default", model)
    features.capability(AssetCapability(_local_runtime_assets(workspace)))
    features.capability(FileSystemCapability(workspace))
    features.capability(ShellCapability())
    features.capability(MemoryCapability())
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(StepPersistenceCapability())
    features.capability(TraceCapability())
    metrics = await _local_metrics(workspace)
    async with Runtime.open(
        "default", storage=_local_runtime_storage(workspace), features=features, metrics=metrics,
        authorization=LocalWorkspaceAuthorization(),
    ) as runtime:
        source = runtime.features
        assert source is not None
        capture = await source.resolve()
        declarations: tuple[dict[str, str | int], ...] = tuple(
            {"kind": item.kind, "id": item.id, "revision": item.revision} for item in capture.contributions
        )
        if not any(item["kind"] == "agent" and item["id"] == "default" for item in declarations):
            declarations = ({"kind": "agent", "id": "default", "revision": 1}, *declarations)
        yield LocalRuntime(runtime, metrics, declarations, {
            "workspace": str(workspace.root), "asset_root": str(workspace.storage_root),
            **({} if status is None else status),
        }, workspace)


def run_local(
    args: Namespace, execute: "Callable[[LocalRuntime], Awaitable[int]]",
) -> int:
    """Run a command while owning its Runtime and local resource lifetime."""
    async def run() -> int:
        workspace = _load_workspace(args.project)
        options = _ModelOptions.load(workspace, args)
        model = options.create_model_binding()
        async with _open_local_runtime(
            workspace, model=model, status=options.status(model),
        ) as local:
            return await execute(local)

    try:
        return asyncio.run(run())
    except (AIError, ImportError, TypeError, ValueError) as error:
        raise CommandError(str(error)) from error


__all__ = ["LocalRuntime", "add_local_runtime_arguments", "local_turn_notices", "run_local"]
