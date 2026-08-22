#!/usr/bin/env python3
from pathlib import Path

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


path = "linktools-ai/src/linktools/ai/agent/_executor.py"
value = read(path)
value = replace_once(value, "from ._definition import AgentDefinition\n", "from ._binding import AgentBinding\nfrom ._definition import AgentDefinition\n", "executor import")
if value.count("        definition: AgentDefinition,\n") != 2:
    raise RuntimeError("executor signatures changed unexpectedly")
value = value.replace("        definition: AgentDefinition,\n", "        binding: AgentBinding,\n", 2)
value = replace_once(value, "        run_usage = RunUsage()\n        usage_limits = _to_usage_limits(definition.spec.usage_limits)", "        definition = binding.definition\n        run_usage = RunUsage()\n        usage_limits = _to_usage_limits(definition.spec.usage_limits)", "executor public derivation")
value = replace_once(value, "                definition,\n                user_prompt,", "                binding,\n                user_prompt,", "executor call")
value = replace_once(value, "    ) -> AgentExecutionResult:\n        if not self._execution_root.is_dir():", "    ) -> AgentExecutionResult:\n        definition = binding.definition\n        if not self._execution_root.is_dir():", "executor private derivation")
value = replace_once(value, "        agent = build_pydantic_agent(definition, model=model)", "        agent = build_pydantic_agent(binding, model=model)", "executor builder")
value = replace_once(value, "        if not isinstance(output, definition.output_type):", "        if not isinstance(output, binding.output_type):", "executor output")
write(path, value)

path = "linktools-ai/src/linktools/ai/asset/_repository.py"
value = read(path)
value = replace_once(value, "    AssetTypeBinding,\n    AssetTypeRegistrySnapshot,", "    AssetTypeBinding,\n    AssetTypeRegistry,", "asset imports")
old = '''class AssetRepository:\n    """Resolve typed logical assets over the raw AssetStore API."""\n\n    def __init__(self, store: AssetStore, registry: AssetTypeRegistrySnapshot) -> None:\n        if store is None or registry is None or not registry.frozen:\n            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n        self._store = store\n        self._registry = registry\n        self._locks = _RepositoryKeyedLock()\n\n    @property\n    def ready(self) -> bool:\n        """Return whether the underlying raw asset store is initialized."""\n        return self._store.ready\n'''
new = '''class AssetRepository:\n    """Resolve typed logical assets over the raw AssetStore API."""\n\n    def __init__(\n        self,\n        store: AssetStore,\n        bindings: Sequence[AssetTypeBinding[object]],\n    ) -> None:\n        if store is None:\n            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n        registry = AssetTypeRegistry()\n        for binding in tuple(bindings):\n            registry.register(binding)\n        self._store = store\n        self._registry = registry.freeze()\n        self._locks = _RepositoryKeyedLock()\n\n    @property\n    def ready(self) -> bool:\n        """Return whether the underlying raw asset store is initialized."""\n        return self._store.ready\n\n    @property\n    def kinds(self) -> tuple[str, ...]:\n        """Return registered logical Asset kinds in canonical order."""\n        return self._registry.kinds\n\n    def binding(self, kind: str) -> AssetTypeBinding[object]:\n        """Return the logical binding registered for one kind."""\n        return self._registry.binding(kind)\n'''
value = replace_once(value, old, new, "asset repository")
write(path, value)

path = "linktools-ai/src/linktools/ai/asset/__init__.py"
value = read(path)
value = value.replace("    AssetTypeRegistry,\n", "").replace("    AssetTypeRegistrySnapshot,\n", "")
value = value.replace('    "AssetTypeRegistry",\n', "").replace('    "AssetTypeRegistrySnapshot",\n', "")
write(path, value)

path = "linktools-ai/src/linktools/ai/workspace/_factory.py"
value = read(path)
value = value.replace("AgentDefinitionCatalog", "AgentCatalog")
value = value.replace("    AssetTypeRegistry,\n", "").replace("    AssetTypeRegistrySnapshot,\n", "")
start = value.index("def _build_asset_registry(")
end = value.index("\ndef _build_providers(", start)
replacement = '''def _merge_asset_bindings(\n    bindings: Sequence[AssetTypeBinding[object]],\n) -> tuple[AssetTypeBinding[object], ...]:\n    selected = {binding.kind: binding for binding in builtin_asset_bindings()}\n    seen: set[str] = set()\n    for binding in bindings:\n        if binding.kind in seen:\n            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT)\n        seen.add(binding.kind)\n        previous = selected.get(binding.kind)\n        if previous is not None and previous.value_type is not binding.value_type:\n            raise AIError(ErrorCode.ASSET_CODEC_CONFLICT)\n        selected[binding.kind] = binding\n    return tuple(selected[kind] for kind in sorted(selected))\n\n\nasync def build_workspace_assets(\n    workspace: Workspace,\n    *,\n    store: AssetStore | None = None,\n    bindings: Sequence[AssetTypeBinding[object]] = (),\n    path_adapter: AssetPathAdapter | None = None,\n) -> AssetRepository:\n    if not isinstance(workspace, Workspace):\n        raise TypeError("workspace is required")\n    effective_bindings = _merge_asset_bindings(bindings)\n    kinds = tuple(binding.kind for binding in effective_bindings)\n    selected_store = store\n    if selected_store is None:\n        selected_adapter = path_adapter\n        if selected_adapter is None:\n            prefixes = {\n                "agent": "agents",\n                "skill": "skills",\n                **{kind: kind for kind in kinds if kind not in {"agent", "skill"}},\n            }\n            selected_adapter = PrefixAssetPathAdapter(prefixes)\n        selected_adapter.validate(kinds)\n        source = DirectoryAssetBackend(str(workspace.storage_root), path_adapter=selected_adapter, kinds=kinds)\n        writable = InMemoryAssetBackend()\n        selected_store = AssetStore(StorageOverlay(source, writer=writable, layers=(StorageLayer("workspace-defaults", writable),)))\n    elif path_adapter is not None:\n        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n    if not selected_store.ready:\n        await selected_store.initialize()\n    return AssetRepository(selected_store, effective_bindings)\n\n'''
value = value[:start] + replacement + value[end + 1:]
value = replace_once(value, "    snapshot: AssetTypeRegistrySnapshot,\n    providers: Sequence[CapabilityProvider],", "    providers: Sequence[CapabilityProvider],", "workspace bind signature")
value = value.replace("for kind in sorted(snapshot.kinds):\n            binding = snapshot.binding(kind)", "for kind in assets.kinds:\n            binding = assets.binding(kind)")
value = replace_once(value, "    assets: AssetRepository,\n    snapshot: AssetTypeRegistrySnapshot,\n) -> \"dict[str, AgentSpec]\":", "    assets: AssetRepository,\n) -> \"dict[str, AgentSpec]\":", "workspace discover signature")
value = value.replace("for kind in sorted(snapshot.kinds):\n        if snapshot.binding(kind).value_type is not AgentSpec:", "for kind in assets.kinds:\n        if assets.binding(kind).value_type is not AgentSpec:")
value = value.replace('    specs.setdefault("default", AgentSpec("default", 1, "default"))', '    specs.setdefault("default", AgentSpec("default"))')
value = value.replace('        "metadata": dict(spec.metadata),\n', "")
open_start = value.index("@asynccontextmanager\nasync def open_workspace_runtime(")
compose_start = value.index("\nasync def _compose_runtime(", open_start)
open_replacement = '''@asynccontextmanager\nasync def open_workspace_runtime(\n    workspace: Workspace,\n    *,\n    tenant_id: str | None = None,\n    assets: AssetRepository | None = None,\n    state: RuntimeState | None = None,\n    models: ModelRegistry | None = None,\n    capability_providers: Sequence[CapabilityProvider] = (),\n    capabilities: Sequence[RuntimeCapability] = (),\n) -> AsyncIterator[Runtime]:\n    if not isinstance(workspace, Workspace):\n        raise TypeError("workspace is required")\n    if any(not isinstance(capability, RuntimeCapability) for capability in capabilities):\n        raise TypeError("capabilities must contain RuntimeCapability values")\n    effective_tenant_id = "default" if tenant_id is None else validate_tenant_id(tenant_id)\n    selected_assets = assets\n    if selected_assets is None:\n        selected_assets = await build_workspace_assets(workspace)\n    elif not isinstance(selected_assets, AssetRepository):\n        raise TypeError("assets must be AssetRepository or None")\n    elif not selected_assets.ready:\n        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n    selected_models = models or _build_default_models(workspace)\n    model_resolver = selected_models.snapshot()\n    providers = _build_providers(capability_providers)\n    initial_revision = await selected_assets.current_revision()\n    asset_capabilities = await _bind_asset_capabilities(selected_assets, providers)\n    runtime_capabilities = tuple(capabilities)\n    platform_capabilities = build_workspace_capabilities(workspace.root)\n    global_capabilities: tuple[CapabilityBinding, ...] = (*asset_capabilities, *runtime_capabilities)\n    specs = await _discover_agent_specs(selected_assets)\n    runtime_fingerprint = _runtime_fingerprint(\n        active_provider_fingerprint=_active_provider_fingerprint(providers, asset_capabilities),\n        agent_catalog_source_fingerprint=_agent_catalog_source_fingerprint(specs, model_resolver),\n        platform_policy_fingerprint=_platform_policy_fingerprint(),\n    )\n    compiler = AgentCompiler(\n        model_resolver=model_resolver, capabilities=global_capabilities,\n        platform_capabilities=platform_capabilities, runtime_fingerprint=runtime_fingerprint,\n        trusted_tool_classes=_trusted_tool_classes(asset_capabilities),\n        trusted_mcp_selectors=_trusted_mcp_selectors(asset_capabilities),\n    )\n    catalog = _build_catalog(specs, compiler)\n    if await selected_assets.current_revision() != initial_revision:\n        raise AIError(ErrorCode.STORAGE_CONFLICT)\n    selected_runtime = state or _default_runtime_state(workspace)\n    try:\n        await selected_runtime.initialize(namespace=workspace.workspace_id, tenant_id=effective_tenant_id)\n        runtime_value = await _compose_runtime(workspace, catalog, compiler=compiler, assets=selected_assets, tenant_id=effective_tenant_id, state=selected_runtime)\n    except BaseException:\n        await selected_runtime.close()\n        raise\n    try:\n        yield runtime_value\n    except BaseException as body_error:\n        try:\n            await runtime_value.close()\n        except BaseException as close_error:\n            raise close_error from body_error\n        raise\n    else:\n        await runtime_value.close()\n\n'''
value = value[:open_start] + open_replacement + value[compose_start + 1:]
value = value.replace('__all__ = ["open_workspace_runtime"]', '__all__ = ["build_workspace_assets", "open_workspace_runtime"]')
write(path, value)

path = "linktools-ai/src/linktools/ai/workspace/__init__.py"
value = read(path)
value = replace_once(value, "from ._factory import open_workspace_runtime", "from ._factory import build_workspace_assets, open_workspace_runtime", "workspace export import")
value = replace_once(value, '    "open_workspace_runtime",\n', '    "build_workspace_assets",\n    "open_workspace_runtime",\n', "workspace export")
write(path, value)

path = "linktools-ai/src/linktools/ai/runtime/_agent.py"
write(path, '''#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n"""Runtime-bound Agent execution handle."""\n\nfrom collections.abc import AsyncIterator, Mapping\nfrom dataclasses import dataclass\nfrom typing import TYPE_CHECKING\nfrom pydantic import BaseModel\nfrom ..core import JsonValue, Principal\nfrom .service_api import EvaluationHandle, ExecutionHandle, ExecutionResult, ExecutionStreamEvent, ReplayEvaluationRequest, RunEvaluationRequest, SessionView\n\nif TYPE_CHECKING:\n    from ..task import TaskNode\n    from ._runtime_service import Runtime\n\n\n@dataclass(frozen=True, slots=True)\nclass AgentHandle:\n    _runtime: "Runtime"\n    agent_id: str\n    _agent_digest: str\n\n    async def start(self, user_prompt: str, *, output: "type[BaseModel] | None" = None, principal: "Principal | None" = None, session_id: "str | None" = None, idempotency_key: "str | None" = None, memory_scope: "str | None" = None, planning: bool = False, thinking: bool = False) -> ExecutionHandle:\n        return await self._runtime._start_for_agent(self._agent_digest, user_prompt, output=output, principal=principal, session_id=session_id, idempotency_key=idempotency_key, memory_scope=memory_scope, planning=planning, thinking=thinking)\n\n    async def run(self, user_prompt: str, *, output: "type[BaseModel] | None" = None, principal: "Principal | None" = None, session_id: "str | None" = None, idempotency_key: "str | None" = None, memory_scope: "str | None" = None, planning: bool = False, thinking: bool = False, timeout_seconds: "float | None" = None) -> ExecutionResult:\n        return await self._runtime._run_for_agent(self._agent_digest, user_prompt, output=output, principal=principal, session_id=session_id, idempotency_key=idempotency_key, memory_scope=memory_scope, planning=planning, thinking=thinking, timeout_seconds=timeout_seconds)\n\n    def stream(self, user_prompt: str, *, output: "type[BaseModel] | None" = None, principal: "Principal | None" = None, session_id: "str | None" = None, idempotency_key: "str | None" = None, memory_scope: "str | None" = None, planning: bool = False, thinking: bool = False) -> AsyncIterator[ExecutionStreamEvent]:\n        return self._runtime._stream_for_agent(self._agent_digest, user_prompt, output=output, principal=principal, session_id=session_id, idempotency_key=idempotency_key, memory_scope=memory_scope, planning=planning, thinking=thinking)\n\n    async def create_session(self, session_id: str, *, principal: "Principal | None" = None, cwd: "str | None" = None, metadata: "Mapping[str, JsonValue] | None" = None) -> SessionView:\n        return await self._runtime._create_session_for_agent(self._agent_digest, session_id, principal=principal, cwd=cwd, metadata=metadata)\n\n    async def run_evaluation(self, request: RunEvaluationRequest, *, output: "type[BaseModel] | None" = None) -> EvaluationHandle:\n        return await self._runtime._run_evaluation_for_agent(self._agent_digest, request, output=output)\n\n    async def replay_evaluation(self, snapshot_id: str, request: ReplayEvaluationRequest, *, output: "type[BaseModel] | None" = None) -> ExecutionHandle:\n        return await self._runtime._replay_evaluation_for_agent(self._agent_digest, snapshot_id, request, output=output)\n\n    def task(self, node_id: str, user_prompt: str, *, dependencies: tuple[str, ...] = (), budget_cost: int = 1, output: "type[BaseModel] | None" = None, planning: bool = False, thinking: bool = False) -> "TaskNode":\n        return self._runtime._task_for_agent(self._agent_digest, node_id, user_prompt, dependencies=dependencies, budget_cost=budget_cost, output=output, planning=planning, thinking=thinking)\n\n\n__all__ = ["AgentHandle"]\n''')

for path in (
    "linktools-ai/src/linktools/ai/runtime/_execution.py",
    "linktools-ai/src/linktools/ai/runtime/_factory.py",
    "linktools-ai/src/linktools/ai/runtime/_local.py",
    "linktools-ai/src/linktools/ai/runtime/_planner.py",
    "linktools-ai/src/linktools/ai/runtime/_runtime_service.py",
    "linktools-ai/src/linktools/ai/runtime/_subagent.py",
):
    write(path, read(path).replace("AgentDefinitionCatalog", "AgentCatalog"))
