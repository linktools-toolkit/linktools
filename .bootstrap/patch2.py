#!/usr/bin/env python3
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

BASE = "a4ded79a152ccfbc635503ee7b7bd394bd50970e"

repo_path = Path("linktools-ai/src/linktools/ai/asset/_repository.py")
repo = repo_path.read_text(encoding="utf-8")
repo = repo.replace(
    "from pathlib import PurePosixPath\n",
    "from pathlib import PurePosixPath\nfrom types import MappingProxyType\n",
    1,
)
repo = repo.replace(
    "    AssetTypeBinding,\n    AssetTypeRegistry,\n    AssetVariantBinding,\n",
    "    AssetTypeBinding,\n    AssetVariantBinding,\n",
    1,
)
repo = repo.replace(
    "    SingleFileLayout,\n)\n",
    "    SingleFileLayout,\n    _validate_layouts,\n)\n",
    1,
)
old_init = '''        if store is None:\n            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n        registry = AssetTypeRegistry()\n        for binding in tuple(bindings):\n            registry.register(binding)\n        self._store = store\n        self._registry = registry.freeze()\n        self._locks = _RepositoryKeyedLock()\n'''
new_init = '''        if store is None:\n            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)\n        selected: dict[str, AssetTypeBinding[object]] = {}\n        for binding in tuple(bindings):\n            if not isinstance(binding, AssetTypeBinding):\n                raise TypeError("bindings must contain AssetTypeBinding values")\n            if binding.kind in selected:\n                raise AIError(ErrorCode.ASSET_CODEC_CONFLICT, "asset kind is already registered")\n            _validate_layouts(binding)\n            selected[binding.kind] = binding\n        self._store = store\n        self._bindings: Mapping[str, AssetTypeBinding[object]] = MappingProxyType(dict(selected))\n        self._kinds = tuple(sorted(selected))\n        self._layout_digest = canonical_sha256(\n            [\n                {\n                    "kind": binding.kind,\n                    "variants": [\n                        {"name": variant.name, "layout": variant.layout.descriptor()}\n                        for variant in sorted(binding.variants, key=lambda item: item.name)\n                    ],\n                    "default_write_variant": binding.default_write_variant,\n                    "allow_nested_id": binding.allow_nested_id,\n                }\n                for binding in sorted(selected.values(), key=lambda item: item.kind)\n            ]\n        )\n        self._locks = _RepositoryKeyedLock()\n'''
if old_init not in repo:
    raise SystemExit("AssetRepository constructor fragment not found")
repo = repo.replace(old_init, new_init, 1)
repo = repo.replace("        return self._registry.kinds\n", "        return self._kinds\n", 1)
old_binding = '''        return self._registry.binding(kind)\n'''
new_binding = '''        try:\n            return self._bindings[kind]\n        except KeyError as error:\n            raise AIError(ErrorCode.ASSET_CODEC_UNKNOWN) from error\n'''
if repo.count(old_binding) != 2:
    raise SystemExit(f"unexpected registry binding usage count: {repo.count(old_binding)}")
repo = repo.replace(old_binding, new_binding, 1)
repo = repo.replace("self._registry.layout_digest", "self._layout_digest")
repo = repo.replace(old_binding, "        return self.binding(kind)\n", 1)
if "_registry" in repo or "AssetTypeRegistry" in repo:
    raise SystemExit("AssetRepository still contains registry state")
repo_path.write_text(repo, encoding="utf-8")

logical_path = Path("linktools-ai/src/linktools/ai/asset/_logical.py")
logical = logical_path.read_text(encoding="utf-8")
logical = logical.replace("from types import MappingProxyType\n", "", 1)
logical = logical.replace("    cast,\n", "", 1)
logical = logical.replace("_SNAPSHOT_TOKEN = object()\n\n", "", 1)
start = logical.index("class AssetTypeRegistry:\n")
end = logical.index("def _validate_layouts(", start)
logical = logical[:start] + logical[end:]
manifest_start = logical.index("def _manifest_entries(")
manifest_end = logical.index("\n\n__all__ = [", manifest_start)
logical = logical[:manifest_start] + logical[manifest_end:]
logical = logical.replace('    "AssetTypeRegistry",\n', "")
logical = logical.replace('    "AssetTypeRegistrySnapshot",\n', "")
if "AssetTypeRegistry" in logical or "RegistrySnapshot" in logical or "manifest_entries" in logical:
    raise SystemExit("logical asset registry residue remains")
logical_path.write_text(logical, encoding="utf-8")

# Final semantic audit.
violations: list[str] = []
source_roots = [Path("linktools-ai/src/linktools/ai"), Path("linktools-ai/src/linktools/commands/ai")]
python_files = [path for root in source_roots for path in root.rglob("*.py")]
for path in python_files:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "agent":
            invalid = {kw.arg for kw in node.keywords if kw.arg in {"output", "planning", "thinking"}}
            if invalid:
                violations.append(f"{path}:{node.lineno}: agent call owns {sorted(invalid)}")
        if isinstance(node, ast.ClassDef) and node.name == "Runtime":
            methods = {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
            stale = methods & {"start", "run", "stream", "create_session", "run_evaluation", "replay_evaluation"}
            if stale:
                violations.append(f"{path}:{node.lineno}: Runtime shortcuts {sorted(stale)}")
        if isinstance(node, ast.ClassDef) and node.name in {"SessionRecord", "SessionView", "ContextProjection"}:
            fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
            if "binding_digest" in fields:
                violations.append(f"{path}:{node.lineno}: {node.name}.binding_digest")
        if isinstance(node, ast.ClassDef) and node.name == "AgentDefinition":
            fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
            stale = fields & {"output_binding", "binding_snapshot", "output_type", "output_schema"}
            if stale:
                violations.append(f"{path}:{node.lineno}: AgentDefinition output state {sorted(stale)}")
        if isinstance(node, ast.ClassDef) and node.name == "AgentHandle":
            fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
            stale = fields & {"planning", "thinking", "binding_digest"}
            if stale:
                violations.append(f"{path}:{node.lineno}: AgentHandle state {sorted(stale)}")

banned = (
    "AgentDefinitionCatalog",
    "AssetTypeRegistry",
    "AssetTypeRegistrySnapshot",
    "asset_bindings=",
    "_AGENT_TASK_LEGACY_V1_FIELDS",
    "_AGENT_TASK_CURRENT_V1_FIELDS",
    "_EXECUTION_LEGACY_V1_FIELDS",
    "_EXECUTION_CURRENT_V1_FIELDS",
    "_EXECUTION_V1_LEGACY_FIELDS",
    "_EXECUTION_V1_CURRENT_FIELDS",
    "_RECOVERY_EXECUTION_V1_LEGACY_FIELDS",
    "_RECOVERY_EXECUTION_V1_CURRENT_FIELDS",
)
for path in [*python_files, Path("linktools-ai/README.md")]:
    text = path.read_text(encoding="utf-8")
    for token in banned:
        if token in text:
            violations.append(f"{path}: stale token {token}")

contracts = "linktools-ai/src/linktools/ai/runtime/state/_contracts.py"
base_contracts = subprocess.check_output(["git", "show", f"{BASE}:{contracts}"], text=True)
current_contracts = Path(contracts).read_text(encoding="utf-8")

def signature(text: str, name: str) -> list[tuple[str, str]]:
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return [
                (item.target.id, ast.unparse(item.annotation))
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ]
    raise RuntimeError(name)

if signature(base_contracts, "ExecutionEventRecord") != signature(current_contracts, "ExecutionEventRecord"):
    violations.append("ExecutionEventRecord declaration drifted unexpectedly")

if violations:
    raise SystemExit("final composition audit failed:\n- " + "\n- ".join(violations))
print("AssetRepository owns its immutable binding index directly")
print("final composition residue audit passed")
print("ExecutionEventRecord declaration unchanged from baseline")
