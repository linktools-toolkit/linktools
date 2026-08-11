#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""`lt ai run`: execute one Asset-backed Agent binding."""

import asyncio
import hashlib
import json
import secrets
import sys
from argparse import Namespace
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandError
from linktools.cli.argparse import ConfigAction
from pydantic_ai.exceptions import ModelAPIError, UserError

from ...ai.agent import (
    ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
    ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
    AssistantTextOutput,
    OutputTypeRegistry,
)
from ...ai.app import (
    BOUND_RUNTIME_PROFILE_FINGERPRINT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    Runtime,
    RuntimePersistenceConfig,
    build_agent_binding_composer,
    build_asset_repository,
    build_asset_store,
    build_local_runtime_services,
    build_runtime,
    open_runtime_resources,
)
from ...ai.asset import (
    AssetInfo,
    AssetKey,
    InMemoryAssetBackend,
    LocalDirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from ...ai.core import ExecutionStatus, JsonValue, TenantAuthorizationPolicy
from ...ai.model import (
    ModelConnectionConfig,
    ModelConnectionRegistry,
    ModelRegistry,
    ModelRoute,
    OpenAIModelMaterializer,
    SnapshotModelResolver,
    StaticModelCredentialProvider,
)
from ...ai.runtime import ExecutionRequest
from ...ai.spec import AgentSpec, AgentSpecCodec, PromptSpec, PromptSpecCodec
from ...ai.storage import StorageLayer, StorageOverlay
from ...ai.workspace import Workspace, trusted_workspace_principal

if TYPE_CHECKING:
    from linktools.cli import CommandParser


class Command(BaseCommand):
    """Run one Asset-backed Agent task locally."""

    @property
    def known_errors(self) -> "list[type[BaseException]]":
        return super().known_errors + [ModelAPIError, UserError]

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("prompt", help="the prompt")
        parser.add_argument("--project", type=Path, default=None, help="working directory")
        parser.add_argument("--base-url", action=ConfigAction, config=OPENAI_BASE_URL)
        parser.add_argument("--model", action=ConfigAction, config=OPENAI_MODEL)
        parser.add_argument("--api-key", action=ConfigAction, config=OPENAI_API_KEY)
        parser.add_argument("--json", action="store_true", help="emit one final JSON result")

    def run(self, args: Namespace) -> int:
        workspace = Workspace.discover(Path.cwd(), root=args.project)
        agent_id = _config_id(None, workspace.config, "default_agent")
        prompt_id = _config_id(None, workspace.config, "default_prompt")
        if not isinstance(args.model, str) or not args.model.strip():
            raise CommandError("--model is required")
        asset_root = (workspace.root / ".linktools").expanduser().resolve()
        runtime_config = RuntimePersistenceConfig.filesystem(
            str(workspace.root),
            workspace_id=workspace.workspace_id,
        )

        async def execute() -> int:
            asset_store = build_asset_store(
                await _build_asset_storage(asset_root)
            )
            await asset_store.initialize()
            assets = build_asset_repository(asset_store)

            output_types = OutputTypeRegistry()
            output_types.register(
                ASSISTANT_TEXT_OUTPUT_SCHEMA_ID,
                ASSISTANT_TEXT_OUTPUT_SCHEMA_REVISION,
                AssistantTextOutput,
            )
            output_types.freeze()

            connection_id = "local-openai"
            credential_id = connection_id + "-api-key" if args.api_key else None
            connection = ModelConnectionConfig(
                connection_id=connection_id,
                base_url=args.base_url,
                credential_id=credential_id,
            )
            connections = ModelConnectionRegistry((connection,))
            routes = ModelRegistry()
            route_snapshot = routes.prime(
                {
                    "default": ModelRoute(
                        route_id="default",
                        provider="openai",
                        model=args.model,
                        connection_id=connection_id,
                    )
                }
            )
            model_resolver = SnapshotModelResolver(route_snapshot)
            credentials = StaticModelCredentialProvider(
                {} if args.api_key is None else {credential_id: args.api_key}
            )
            materializer = OpenAIModelMaterializer(credentials)
            composer = build_agent_binding_composer(
                assets,
                model_resolver=model_resolver,
                model_connections=connections,
                output_types=output_types,
                execution_profile_fingerprint=BOUND_RUNTIME_PROFILE_FINGERPRINT,
            )
            binding = await composer.compose(agent_id=agent_id, prompt_id=prompt_id)
            principal = trusted_workspace_principal(workspace.workspace_id)
            request = ExecutionRequest(
                prompt=args.prompt,
                principal=principal,
                idempotency_key=secrets.token_urlsafe(32),
                memory_namespace=workspace.workspace_id,
            )
            authorization = TenantAuthorizationPolicy()
            grant_key = hashlib.sha256(f"workspace:{workspace.workspace_id}".encode()).digest()
            async with open_runtime_resources(runtime_config) as resources:
                event_printer = _CliEventPrinter()
                local = build_local_runtime_services(
                    resources,
                    authorization,
                    grant_key=grant_key,
                    materializer=materializer,
                    execution_root=workspace.root,
                    event_handler=event_printer,
                )
                runtime = build_runtime(binding, local=local)
                return await _emit_result(runtime, request, args.json)

        try:
            return asyncio.run(execute())
        except (TypeError, ValueError) as error:
            raise CommandError(str(error)) from error


async def _build_asset_storage(root: Path) -> "StorageOverlay[AssetKey, bytes, AssetInfo]":
    local = LocalDirectoryAssetBackend(
        str(root),
        writable=False,
        path_adapter=PrefixAssetPathAdapter({"skill": "skills"}),
    )
    defaults = InMemoryAssetBackend()
    await defaults.put(
        AssetKey("agent", "default"),
        AgentSpecCodec().encode(AgentSpec("default", 1, "default", (), "assistant-text", 1)),
    )
    await defaults.put(
        AssetKey("prompt", "default"),
        PromptSpecCodec().encode(PromptSpec("default", 1, "", ())),
    )
    return StorageOverlay(local, layers=(StorageLayer("defaults", defaults),))


async def _emit_result(runtime: Runtime, request: ExecutionRequest, as_json: bool) -> int:
    try:
        result = await runtime.run(request)
    finally:
        _CliEventPrinter.finish_active_output()
    succeeded = result.status is ExecutionStatus.SUCCEEDED
    payload = {
        "execution_id": result.execution_id,
        "status": result.status.value,
        "output": result.output if succeeded else None,
        "output_schema_id": result.output_schema_id,
        "output_schema_revision": result.output_schema_revision,
        "output_schema_fingerprint": result.output_schema_fingerprint,
    }
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    if not succeeded:
        raise CommandError(f"execution {result.execution_id} finished with status {result.status.value}")
    if not isinstance(result.output, dict) or not isinstance(result.output.get("text"), str):
        raise CommandError("assistant-text output is invalid")
    if not as_json:
        print(result.output["text"])
    return 0


class _CliEventPrinter:
    _active: "_CliEventPrinter | None" = None

    def __init__(self) -> None:
        self._thinking = False
        type(self)._active = self

    def __call__(self, event: "dict[str, JsonValue]") -> None:
        event_type = event.get("type")
        if event_type == "thinking":
            if not self._thinking:
                sys.stderr.write("\n[thinking] ")
                self._thinking = True
            sys.stderr.write(str(event.get("text", "")))
            sys.stderr.flush()
            return
        if event_type == "tool":
            self._finish_thinking()
            phase = event.get("phase")
            name = str(event.get("name", "tool"))
            if phase == "start":
                sys.stderr.write(f"\n[tool] {name} started\n")
            else:
                sys.stderr.write(f"[tool] {name} {str(event.get('status', 'UNKNOWN')).lower()}\n")
            sys.stderr.flush()

    @classmethod
    def finish_active_output(cls) -> None:
        if cls._active is not None:
            cls._active._finish_thinking()

    def _finish_thinking(self) -> None:
        if self._thinking:
            sys.stderr.write("\n")
            sys.stderr.flush()
            self._thinking = False


def _config_id(explicit: object, config: dict[str, object], name: str) -> str:
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    configured = config.get(name)
    return configured if isinstance(configured, str) and configured.strip() else "default"


command = Command()
