#!/usr/bin/env python3
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

BASE = "a4ded79a152ccfbc635503ee7b7bd394bd50970e"


def replace_exact(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError(f"expected fragment not found: {path}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


replace_exact(
    "tests/ai/test_session_admission_conformance.py",
    '''def _binding(digest: str) -> AgentBindingSnapshot:\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        agent_digest="d" * 64,\n        output_type_module="builtins",\n        output_type_qualname="str",\n        output_schema_id="test-output",\n        output_schema_revision=1,\n        output_schema_fingerprint="c" * 64,\n        local_runtime_capability_descriptors=(),\n        binding_digest=digest,\n    )\n\n\nclass _DefinitionCatalog:\n    def binding(self, digest: str) -> object:\n        return SimpleNamespace(digest=digest, snapshot=_binding(digest))\n''',
    '''def _binding(digest: str) -> AgentBindingSnapshot:\n    return AgentBindingSnapshot(\n        version=1,\n        agent_spec=AgentSpec("agent", 1, "model"),\n        agent_digest="b" * 64,\n        output_type_module="builtins",\n        output_type_qualname="str",\n        output_schema_id="test-output",\n        output_schema_revision=1,\n        output_schema_fingerprint="c" * 64,\n        local_runtime_capability_descriptors=(),\n        binding_digest=digest,\n    )\n\n\nclass _DefinitionCatalog:\n    def binding(self, digest: str) -> object:\n        return SimpleNamespace(\n            digest=digest,\n            definition=SimpleNamespace(digest="b" * 64),\n            snapshot=_binding(digest),\n        )\n''',
)

replace_exact(
    "tests/ai/test_session_admission_conformance.py",
    '''            sessions=state.conversation.sessions,\n        catalog=_DefinitionCatalog(),\n        compiler=object(),\n            backend=backend,\n''',
    '''            sessions=state.conversation.sessions,\n            catalog=_DefinitionCatalog(),\n            compiler=object(),\n            backend=backend,\n''',
)

acceptance_path = Path("tests/ai/test_agent_composition.py")
acceptance = acceptance_path.read_text(encoding="utf-8")
marker = "test_agent_and_binding_identity_split_acceptance"
if marker not in acceptance:
    acceptance += '''\n\n@pytest.mark.asyncio\nasync def test_agent_and_binding_identity_split_acceptance(tmp_path) -> None:\n    from pydantic import BaseModel\n    from linktools.ai.model import ModelRegistry\n    from linktools.ai.workspace import Workspace, open_workspace_runtime\n\n    workspace = Workspace.load(tmp_path)\n    models = ModelRegistry.openai(model="gpt-test")\n    async with open_workspace_runtime(workspace, models=models) as runtime:\n        base = runtime.agent()\n        text_binding = runtime._bind_agent(base._agent_digest)\n        structured_binding = runtime._bind_agent(\n            base._agent_digest,\n            output=BaseModel,\n        )\n\n        assert text_binding.definition.digest == structured_binding.definition.digest\n        assert text_binding.digest != structured_binding.digest\n        assert runtime._bind_agent(base._agent_digest).digest == text_binding.digest\n\n        local_capability = RuntimeCapability.from_spec(\n            "local-identity",\n            _DurableCapability,\n            config={"mode": "strict"},\n            revision=1,\n        )\n        local = runtime.agent(capabilities=(local_capability,))\n        local_binding = runtime._bind_agent(local._agent_digest)\n        assert local._agent_digest != base._agent_digest\n        assert local_binding.digest != text_binding.digest\n\n        session = await base.create_session("identity-session")\n        assert session.agent_digest == base._agent_digest\n        await runtime._ensure_session(\n            runtime._definition(base._agent_digest),\n            session.session_id,\n            runtime.default_principal,\n        )\n        with pytest.raises(AIError) as error:\n            await runtime._ensure_session(\n                runtime._definition(local._agent_digest),\n                session.session_id,\n                runtime.default_principal,\n            )\n        assert error.value.code is ErrorCode.SESSION_BINDING_MISMATCH\n'''
    acceptance_path.write_text(acceptance, encoding="utf-8")


# Mechanical residue audit over the final public/runtime source and README.
violations: list[str] = []
source_roots = [
    Path("linktools-ai/src/linktools/ai"),
    Path("linktools-ai/src/linktools/commands/ai"),
]
python_files = [path for root in source_roots for path in root.rglob("*.py")]

for path in python_files:
    text = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as error:
        violations.append(f"{path}: syntax error during audit: {error}")
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "agent":
            invalid = {kw.arg for kw in node.keywords if kw.arg in {"output", "planning", "thinking"}}
            if invalid:
                violations.append(f"{path}:{node.lineno}: Runtime.agent-style call owns {sorted(invalid)}")
        if isinstance(node, ast.ClassDef) and node.name == "Runtime":
            methods = {item.name for item in node.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}
            forbidden = methods & {"start", "run", "stream", "create_session", "run_evaluation", "replay_evaluation"}
            if forbidden:
                violations.append(f"{path}:{node.lineno}: Runtime exposes forbidden shortcuts {sorted(forbidden)}")
        if isinstance(node, ast.ClassDef) and node.name in {"SessionRecord", "SessionView", "ContextProjection"}:
            fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
            if "binding_digest" in fields:
                violations.append(f"{path}:{node.lineno}: {node.name} still owns binding_digest")
        if isinstance(node, ast.ClassDef) and node.name == "AgentDefinition":
            fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
            stale = fields & {"output_binding", "binding_snapshot", "output_type", "output_schema"}
            if stale:
                violations.append(f"{path}:{node.lineno}: AgentDefinition owns output state {sorted(stale)}")
        if isinstance(node, ast.ClassDef) and node.name == "AgentHandle":
            fields = {item.target.id for item in node.body if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)}
            stale = fields & {"planning", "thinking", "binding_digest"}
            if stale:
                violations.append(f"{path}:{node.lineno}: AgentHandle owns execution/binding state {sorted(stale)}")

banned_tokens = {
    "AgentDefinitionCatalog": "obsolete definition catalog",
    "AssetTypeRegistry": "obsolete public asset registry",
    "asset_bindings=": "obsolete workspace asset_bindings API",
    "_AGENT_TASK_LEGACY_V1_FIELDS": "legacy TaskGraph V1 branch",
    "_AGENT_TASK_CURRENT_V1_FIELDS": "dual TaskGraph V1 branch",
    "_EXECUTION_LEGACY_V1_FIELDS": "legacy Temporal execution V1 branch",
    "_EXECUTION_CURRENT_V1_FIELDS": "dual Temporal execution V1 branch",
    "_EXECUTION_V1_LEGACY_FIELDS": "legacy RuntimeState execution V1 branch",
    "_EXECUTION_V1_CURRENT_FIELDS": "dual RuntimeState execution V1 branch",
    "_RECOVERY_EXECUTION_V1_LEGACY_FIELDS": "legacy recovery V1 branch",
    "_RECOVERY_EXECUTION_V1_CURRENT_FIELDS": "dual recovery V1 branch",
}
for path in [*python_files, Path("linktools-ai/README.md")]:
    text = path.read_text(encoding="utf-8")
    for token, reason in banned_tokens.items():
        if token in text:
            violations.append(f"{path}: {reason}: {token}")

# ExecutionEventRecord itself must not have drifted during the identity refactor.
contracts = "linktools-ai/src/linktools/ai/runtime/state/_contracts.py"
base_contracts = subprocess.check_output(["git", "show", f"{BASE}:{contracts}"], text=True)
current_contracts = Path(contracts).read_text(encoding="utf-8")


def class_signature(text: str, name: str) -> list[tuple[str, str]]:
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return [
                (item.target.id, ast.unparse(item.annotation))
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            ]
    raise RuntimeError(f"class not found: {name}")

base_event = class_signature(base_contracts, "ExecutionEventRecord")
current_event = class_signature(current_contracts, "ExecutionEventRecord")
if current_event != base_event:
    violations.append(
        "ExecutionEventRecord declaration drifted unexpectedly: "
        f"base={base_event!r} current={current_event!r}"
    )
else:
    print(f"ExecutionEventRecord declaration unchanged: {current_event!r}")

if violations:
    raise SystemExit("final composition audit failed:\n- " + "\n- ".join(violations))

print("final composition residue audit passed")
print("final exact identity fixture and acceptance coverage applied")
