#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path.cwd()


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, value: str) -> None:
    (ROOT / path).write_text(value, encoding="utf-8")


def replace_once(value: str, old: str, new: str, label: str) -> str:
    count = value.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one match, got {count}")
    return value.replace(old, new, 1)


# AgentExecutor executes the exact AgentBinding, not an output-free definition.
path = "linktools-ai/src/linktools/ai/agent/_executor.py"
value = read(path)
value = replace_once(
    value,
    "from ._definition import AgentDefinition\n",
    "from ._binding import AgentBinding\nfrom ._definition import AgentDefinition\n",
    "executor import",
)
value = replace_once(
    value,
    "        definition: AgentDefinition,\n        user_prompt: str,",
    "        binding: AgentBinding,\n        user_prompt: str,",
    "executor public signature",
)
value = replace_once(
    value,
    "        run_usage = RunUsage()\n        usage_limits = _to_usage_limits(definition.spec.usage_limits)",
    "        definition = binding.definition\n        run_usage = RunUsage()\n        usage_limits = _to_usage_limits(definition.spec.usage_limits)",
    "executor definition derivation",
)
value = replace_once(
    value,
    "                definition,\n                user_prompt,",
    "                binding,\n                user_prompt,",
    "executor internal call",
)
value = replace_once(
    value,
    "        definition: AgentDefinition,\n        user_prompt: str,\n        history:",
    "        binding: AgentBinding,\n        user_prompt: str,\n        history:",
    "executor private signature",
)
value = replace_once(
    value,
    "        if not self._execution_root.is_dir():\n            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)",
    "        definition = binding.definition\n        if not self._execution_root.is_dir():\n            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)",
    "executor private definition",
)
value = replace_once(
    value,
    "        agent = build_pydantic_agent(definition, model=model)",
    "        agent = build_pydantic_agent(binding, model=model)",
    "executor builder",
)
value = replace_once(
    value,
    "        if not isinstance(output, definition.output_type):",
    "        if not isinstance(output, binding.output_type):",
    "executor output validation",
)
write(path, value)

# AssetRepository owns its registry construction and exposes only the typed view Runtime needs.
path = "linktools-ai/src/linktools/ai/asset/_repository.py"
value = read(path)
value = replace_once(
    value,
    "    AssetTypeBinding,\n    AssetTypeRegistrySnapshot,",
    "    AssetTypeBinding,\n    AssetTypeRegistry,",
    "asset logical imports",
)
old = '''class AssetRepository:\n    """Resolve typed logical assets over the raw AssetStore API."""\n\n    def __init__(self, store: AssetStore, registry: AssetTypeRegistrySnapshot) -> None:\n        if store is None or registry is None or not registry.frozen:\n            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n        self._store = store\n        self._registry = registry\n        self._locks = _RepositoryKeyedLock()\n\n    @property\n    def ready(self) -> bool:\n        """Return whether the underlying raw asset store is initialized."""\n        return self._store.ready\n'''
new = '''class AssetRepository:\n    """Resolve typed logical assets over the raw AssetStore API."""\n\n    def __init__(\n        self,\n        store: AssetStore,\n        bindings: Sequence[AssetTypeBinding[object]],\n    ) -> None:\n        if store is None:\n            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n        registry = AssetTypeRegistry()\n        for binding in tuple(bindings):\n            registry.register(binding)\n        self._store = store\n        self._registry = registry.freeze()\n        self._locks = _RepositoryKeyedLock()\n\n    @property\n    def ready(self) -> bool:\n        """Return whether the underlying raw asset store is initialized."""\n        return self._store.ready\n\n    @property\n    def kinds(self) -> tuple[str, ...]:\n        """Return registered logical Asset kinds in canonical order."""\n        return self._registry.kinds\n\n    def binding(self, kind: str) -> AssetTypeBinding[object]:\n        """Return the logical binding registered for one kind."""\n        return self._registry.binding(kind)\n'''
value = replace_once(value, old, new, "asset repository constructor")
write(path, value)

path = "linktools-ai/src/linktools/ai/asset/__init__.py"
value = read(path)
value = value.replace("    AssetTypeRegistry,\n", "").replace("    AssetTypeRegistrySnapshot,\n", "")
value = value.replace('    "AssetTypeRegistry",\n', "").replace('    "AssetTypeRegistrySnapshot",\n', "")
write(path, value)

# Workspace owns optional raw Asset construction; Runtime receives a complete repository.
path = "linktools-ai/src/linktools/ai/workspace/_factory.py"
value = read(path)
value = value.replace("    AssetTypeRegistry,\n", "").replace("    AssetTypeRegistrySnapshot,\n", "")
start = value.index("def _build_asset_registry(")
end = value.index("\ndef _build_providers(", start)
replacement = '''def _merge_asset_bindings(\n    bindings: Sequence[AssetTypeBinding[object]],\n) -> tuple[AssetTypeBinding[object], ...]:\n    selected = {binding.kind: binding for binding in builtin_asset_bindings()}\n    seen: set[str] = set()\n    for binding in bindings:\n        if binding.kind in seen:\n            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT)\n        seen.add(binding.kind)\n        previous = selected.get(binding.kind)\n        if previous is not None and previous.value_type is not binding.value_type:\n            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT)\n        selected[binding.kind] = binding\n    return tuple(selected[kind] for kind in sorted(selected))\n\n\nasync def build_workspace_assets(\n    workspace: Workspace,\n    *,\n    store: AssetStore | None = None,\n    bindings: Sequence[AssetTypeBinding[object]] = (),\n    path_adapter: AssetPathAdapter | None = None,\n) -> AssetRepository:\n    if not isinstance(workspace, Workspace):\n        raise TypeError("workspace is required")\n    effective_bindings = _merge_asset_bindings(bindings)\n    kinds = tuple(binding.kind for binding in effective_bindings)\n    selected_store = store\n    if selected_store is None:\n        selected_adapter = path_adapter\n        if selected_adapter is None:\n            prefixes = {\n                "agent": "agents",\n                "skill": "skills",\n                **{kind: kind for kind in kinds if kind not in {"agent", "skill"}},\n            }\n            selected_adapter = PrefixAssetPathAdapter(prefixes)\n        selected_adapter.validate(kinds)\n        source = DirectoryAssetBackend(\n            str(workspace.storage_root),\n            path_adapter=selected_adapter,\n            kinds=kinds,\n        )\n        writable = InMemoryAssetBackend()\n        selected_store = AssetStore(\n            StorageOverlay(\n                source,\n                writer=writable,\n                layers=(StorageLayer("workspace-defaults", writable),),\n            )\n        )\n    elif path_adapter is not None:\n        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n    if not selected_store.ready:\n        await selected_store.initialize()\n    return AssetRepository(selected_store, effective_bindings)\n\n'''
value = value[:start] + replacement + value[end + 1:]
value = replace_once(
    value,
    '''async def _bind_asset_capabilities(\n    assets: AssetRepository,\n    snapshot: AssetTypeRegistrySnapshot,\n    providers: Sequence[CapabilityProvider],\n) -> "tuple[CapabilityBinding, ...]":''',
    '''async def _bind_asset_capabilities(\n    assets: AssetRepository,\n    providers: Sequence[CapabilityProvider],\n) -> "tuple[CapabilityBinding, ...]":''',
    "workspace bind signature",
)
value = value.replace("for kind in sorted(snapshot.kinds):\n            binding = snapshot.binding(kind)", "for kind in assets.kinds:\n            binding = assets.binding(kind)")
value = replace_once(
    value,
    '''async def _discover_agent_specs(\n    assets: AssetRepository,\n    snapshot: AssetTypeRegistrySnapshot,\n) -> "dict[str, AgentSpec]":''',
    '''async def _discover_agent_specs(\n    assets: AssetRepository,\n) -> "dict[str, AgentSpec]":''',
    "workspace discover signature",
)
value = value.replace("for kind in sorted(snapshot.kinds):\n        if snapshot.binding(kind).value_type is not AgentSpec:", "for kind in assets.kinds:\n        if assets.binding(kind).value_type is not AgentSpec:")
value = value.replace('    specs.setdefault("default", AgentSpec("default", 1, "default"))', '    specs.setdefault("default", AgentSpec("default"))')
value = value.replace('        "metadata": dict(spec.metadata),\n', "")
open_start = value.index("@asynccontextmanager\nasync def open_workspace_runtime(")
compose_start = value.index("\nasync def _compose_runtime(", open_start)
open_replacement = '''@asynccontextmanager\nasync def open_workspace_runtime(\n    workspace: Workspace,\n    *,\n    tenant_id: str | None = None,\n    assets: AssetRepository | None = None,\n    state: RuntimeState | None = None,\n    models: ModelRegistry | None = None,\n    capability_providers: Sequence[CapabilityProvider] = (),\n    capabilities: Sequence[RuntimeCapability] = (),\n) -> AsyncIterator[Runtime]:\n    if not isinstance(workspace, Workspace):\n        raise TypeError("workspace is required")\n    if any(not isinstance(capability, RuntimeCapability) for capability in capabilities):\n        raise TypeError("capabilities must contain RuntimeCapability values")\n    effective_tenant_id = "default" if tenant_id is None else validate_tenant_id(tenant_id)\n    selected_assets = assets\n    if selected_assets is None:\n        selected_assets = await build_workspace_assets(workspace)\n    elif not isinstance(selected_assets, AssetRepository):\n        raise TypeError("assets must be AssetRepository or None")\n    elif not selected_assets.ready:\n        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n    selected_models = models or _build_default_models(workspace)\n    model_resolver = selected_models.snapshot()\n    providers = _build_providers(capability_providers)\n    initial_revision = await selected_assets.current_revision()\n    asset_capabilities = await _bind_asset_capabilities(selected_assets, providers)\n    runtime_capabilities = tuple(capabilities)\n    platform_capabilities = build_workspace_capabilities(workspace.root)\n    global_capabilities: tuple[CapabilityBinding, ...] = (\n        *asset_capabilities,\n        *runtime_capabilities,\n    )\n    specs = await _discover_agent_specs(selected_assets)\n    catalog_fingerprint = _agent_catalog_source_fingerprint(specs, model_resolver)\n    active_provider_fingerprint = _active_provider_fingerprint(providers, asset_capabilities)\n    platform_policy_fingerprint = _platform_policy_fingerprint()\n    runtime_fingerprint = _runtime_fingerprint(\n        active_provider_fingerprint=active_provider_fingerprint,\n        agent_catalog_source_fingerprint=catalog_fingerprint,\n        platform_policy_fingerprint=platform_policy_fingerprint,\n    )\n    compiler = AgentCompiler(\n        model_resolver=model_resolver,\n        capabilities=global_capabilities,\n        platform_capabilities=platform_capabilities,\n        runtime_fingerprint=runtime_fingerprint,\n        trusted_tool_classes=_trusted_tool_classes(asset_capabilities),\n        trusted_mcp_selectors=_trusted_mcp_selectors(asset_capabilities),\n    )\n    catalog = _build_catalog(specs, compiler)\n    if await selected_assets.current_revision() != initial_revision:\n        raise AIError(ErrorCode.STORAGE_CONFLICT)\n    selected_runtime = state or _default_runtime_state(workspace)\n    try:\n        await selected_runtime.initialize(namespace=workspace.workspace_id, tenant_id=effective_tenant_id)\n        runtime_value = await _compose_runtime(\n            workspace,\n            catalog,\n            compiler=compiler,\n            assets=selected_assets,\n            tenant_id=effective_tenant_id,\n            state=selected_runtime,\n        )\n    except BaseException:\n        await selected_runtime.close()\n        raise\n    _logger.info(\n        "workspace Runtime opened: workspace=%s tenant=%s active_providers=%s capabilities=%s agents=%s",\n        workspace.workspace_id,\n        effective_tenant_id,\n        tuple(binding.provider for binding in asset_capabilities),\n        tuple(\n            (capability.provider, capability.id)\n            for capability in (*global_capabilities, *platform_capabilities)\n        ),\n        catalog.root_ids,\n    )\n    try:\n        yield runtime_value\n    except BaseException as body_error:\n        try:\n            await runtime_value.close()\n        except BaseException as close_error:\n            raise close_error from body_error\n        raise\n    else:\n        await runtime_value.close()\n\n'''
value = value[:open_start] + open_replacement + value[compose_start + 1:]
value = value.replace('__all__ = ["open_workspace_runtime"]', '__all__ = ["build_workspace_assets", "open_workspace_runtime"]')
write(path, value)

path = "linktools-ai/src/linktools/ai/workspace/__init__.py"
value = read(path)
value = replace_once(value, "from ._factory import open_workspace_runtime", "from ._factory import build_workspace_assets, open_workspace_runtime", "workspace exports import")
value = replace_once(value, '    "open_workspace_runtime",\n', '    "build_workspace_assets",\n    "open_workspace_runtime",\n', "workspace exports")
write(path, value)

# AgentHandle becomes an Agent-identity handle; output/modes are per execution.
path = "linktools-ai/src/linktools/ai/runtime/_agent.py"
write(path, '''#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Runtime-bound Agent execution handle."""\n\nfrom collections.abc import AsyncIterator, Mapping\nfrom dataclasses import dataclass\nfrom typing import TYPE_CHECKING\n\nfrom pydantic import BaseModel\n\nfrom ..core import JsonValue, Principal\nfrom .service_api import (\n    EvaluationHandle,\n    ExecutionHandle,\n    ExecutionResult,\n    ExecutionStreamEvent,\n    ReplayEvaluationRequest,\n    RunEvaluationRequest,\n    SessionView,\n)\n\nif TYPE_CHECKING:\n    from ..task import TaskNode\n    from ._runtime_service import Runtime\n\n\n@dataclass(frozen=True, slots=True)\nclass AgentHandle:\n    _runtime: "Runtime"\n    agent_id: str\n    _agent_digest: str\n\n    async def start(\n        self,\n        user_prompt: str,\n        *,\n        output: "type[BaseModel] | None" = None,\n        principal: "Principal | None" = None,\n        session_id: "str | None" = None,\n        idempotency_key: "str | None" = None,\n        memory_scope: "str | None" = None,\n        planning: bool = False,\n        thinking: bool = False,\n    ) -> ExecutionHandle:\n        return await self._runtime._start_for_agent(\n            self._agent_digest,\n            user_prompt,\n            output=output,\n            principal=principal,\n            session_id=session_id,\n            idempotency_key=idempotency_key,\n            memory_scope=memory_scope,\n            planning=planning,\n            thinking=thinking,\n        )\n\n    async def run(\n        self,\n        user_prompt: str,\n        *,\n        output: "type[BaseModel] | None" = None,\n        principal: "Principal | None" = None,\n        session_id: "str | None" = None,\n        idempotency_key: "str | None" = None,\n        memory_scope: "str | None" = None,\n        planning: bool = False,\n        thinking: bool = False,\n        timeout_seconds: "float | None" = None,\n    ) -> ExecutionResult:\n        return await self._runtime._run_for_agent(\n            self._agent_digest,\n            user_prompt,\n            output=output,\n            principal=principal,\n            session_id=session_id,\n            idempotency_key=idempotency_key,\n            memory_scope=memory_scope,\n            planning=planning,\n            thinking=thinking,\n            timeout_seconds=timeout_seconds,\n        )\n\n    def stream(\n        self,\n        user_prompt: str,\n        *,\n        output: "type[BaseModel] | None" = None,\n        principal: "Principal | None" = None,\n        session_id: "str | None" = None,\n        idempotency_key: "str | None" = None,\n        memory_scope: "str | None" = None,\n        planning: bool = False,\n        thinking: bool = False,\n    ) -> AsyncIterator[ExecutionStreamEvent]:\n        return self._runtime._stream_for_agent(\n            self._agent_digest,\n            user_prompt,\n            output=output,\n            principal=principal,\n            session_id=session_id,\n            idempotency_key=idempotency_key,\n            memory_scope=memory_scope,\n            planning=planning,\n            thinking=thinking,\n        )\n\n    async def create_session(\n        self,\n        session_id: str,\n        *,\n        principal: "Principal | None" = None,\n        cwd: "str | None" = None,\n        metadata: "Mapping[str, JsonValue] | None" = None,\n    ) -> SessionView:\n        return await self._runtime._create_session_for_agent(\n            self._agent_digest,\n            session_id,\n            principal=principal,\n            cwd=cwd,\n            metadata=metadata,\n        )\n\n    async def run_evaluation(\n        self,\n        request: RunEvaluationRequest,\n        *,\n        output: "type[BaseModel] | None" = None,\n    ) -> EvaluationHandle:\n        return await self._runtime._run_evaluation_for_agent(\n            self._agent_digest, request, output=output\n        )\n\n    async def replay_evaluation(\n        self,\n        snapshot_id: str,\n        request: ReplayEvaluationRequest,\n        *,\n        output: "type[BaseModel] | None" = None,\n    ) -> ExecutionHandle:\n        return await self._runtime._replay_evaluation_for_agent(\n            self._agent_digest, snapshot_id, request, output=output\n        )\n\n    def task(\n        self,\n        node_id: str,\n        user_prompt: str,\n        *,\n        dependencies: tuple[str, ...] = (),\n        budget_cost: int = 1,\n        output: "type[BaseModel] | None" = None,\n        planning: bool = False,\n        thinking: bool = False,\n    ) -> "TaskNode":\n        return self._runtime._task_for_agent(\n            self._agent_digest,\n            node_id,\n            user_prompt,\n            dependencies=dependencies,\n            budget_cost=budget_cost,\n            output=output,\n            planning=planning,\n            thinking=thinking,\n        )\n\n\n__all__ = ["AgentHandle"]\n''')
