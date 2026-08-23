from __future__ import annotations

import ast
import inspect
from dataclasses import fields
from pathlib import Path

import linktools.ai.asset as asset_api
from linktools.ai.agent import AgentBinding, AgentDefinition
from linktools.ai.runtime._agent import AgentHandle
from linktools.ai.runtime._runtime_service import Runtime
from linktools.ai.runtime.state._codec import (
    _V1_CODEC,
    _V1_SCHEMA_FINGERPRINTS,
    _dataclass_schema_fingerprint,
)
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    ExecutionEventRecord,
    SessionRecord,
)

root = Path('linktools-ai')
src = root / 'src/linktools/ai'

# Public Runtime boundary.
sig = inspect.signature(Runtime.agent)
params = list(sig.parameters.values())
assert [p.name for p in params] == ['self', 'agent', 'capabilities'], sig
assert params[2].kind is inspect.Parameter.KEYWORD_ONLY
for name in (
    'start',
    'run',
    'stream',
    'create_session',
    'run_evaluation',
    'replay_evaluation',
):
    assert name not in Runtime.__dict__, f'Runtime shortcut leaked: {name}'

# Identity ownership.
assert [f.name for f in fields(AgentHandle)] == ['_runtime', 'agent_id', '_agent_digest']
assert 'binding_digest' not in {f.name for f in fields(SessionRecord)}
assert 'agent_digest' in {f.name for f in fields(SessionRecord)}
assert 'binding_digest' not in {f.name for f in fields(ContextProjection)}
assert 'agent_digest' in {f.name for f in fields(ContextProjection)}
definition_fields = {f.name for f in fields(AgentDefinition)}
assert not ({'output_binding', 'binding_snapshot', 'output_type', 'output_schema'} & definition_fields)
assert {'digest', 'definition', 'output_binding', 'snapshot'} == {f.name for f in fields(AgentBinding)}

# Asset registry is not a public composition concept.
assert 'AssetTypeRegistry' not in getattr(asset_api, '__all__', ())
assert not hasattr(asset_api, 'AssetTypeRegistry')

# These records use the generic V1 schema descriptor and are the identity/fingerprint
# contracts touched by this refactor. Exact-binding records use custom codecs and
# are validated through their dedicated round-trip tests instead.
for wire_id, target in (
    ('session_record', SessionRecord),
    ('context_projection', ContextProjection),
    ('execution_event', ExecutionEventRecord),
):
    expected = _V1_SCHEMA_FINGERPRINTS[wire_id]
    actual = _dataclass_schema_fingerprint(target, _V1_CODEC)
    assert actual == expected, f'stale schema fingerprint {wire_id}: {expected} != {actual}'

# No removed architecture branches or public aliases may remain in production/docs.
forbidden_tokens = (
    'AgentDefinitionCatalog',
    '_AGENT_TASK_LEGACY_V1_FIELDS',
    '_AGENT_TASK_CURRENT_V1_FIELDS',
    '_EXECUTION_LEGACY_V1_FIELDS',
    '_EXECUTION_CURRENT_V1_FIELDS',
    '_EXECUTION_V1_LEGACY_FIELDS',
    '_RECOVERY_EXECUTION_V1_LEGACY_FIELDS',
)
for path in src.rglob('*.py'):
    text = path.read_text(encoding='utf-8')
    for token in forbidden_tokens:
        assert token not in text, f'{path}: removed token remains: {token}'
    assert 'existing_target.binding_digest' not in text, f'{path}: stale Session identity reference'

# Runtime.agent(...) owns only agent selection/capabilities; output and modes are per call.
def root_name(node: ast.AST) -> str | None:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None

for path in src.rglob('*.py'):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != 'agent' or root_name(node.func.value) not in {'runtime', 'rt'}:
            continue
        bad = {kw.arg for kw in node.keywords if kw.arg in {'output', 'planning', 'thinking'}}
        assert not bad, f'{path}:{node.lineno}: Runtime.agent has execution kwargs {sorted(bad)}'

readme = (root / 'README.md').read_text(encoding='utf-8')
assert 'asset_bindings=' not in readme
assert 'AssetTypeRegistry' not in readme
assert 'legacy root-only V1' not in readme
assert 'legacy/final V1' not in readme

print('final architecture audit passed')
