from pathlib import Path

root = Path('.')

# ExecutionEventRecord and all of its schema inputs are unchanged from the
# baseline. Restore its original V1 fingerprint; the previous repin was caused
# by a broad mechanical update, not by a contract change.
path = root / 'linktools-ai/src/linktools/ai/runtime/state/_codec.py'
text = path.read_text(encoding='utf-8')
old = '"execution_event": "0c977a9bc5de198667dc34640acd192fe54a986227d886ffb1dea596a851c957"'
new = '"execution_event": "c7d10fa9a15092e7e29c503b938f12358b836ee0d578bd5faff0f0951d3bedf8"'
if text.count(old) != 1:
    raise SystemExit(f'execution_event fingerprint anchor count={text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')

# Clear the two pre-existing unused imports exposed by the full final Ruff gate.
path = root / 'linktools-ai/src/linktools/ai/adapter/_history.py'
text = path.read_text(encoding='utf-8')
if '    ContinuableSnapshot,\n' in text:
    text = text.replace('    ContinuableSnapshot,\n', '', 1)
path.write_text(text, encoding='utf-8')

path = root / 'linktools-ai/src/linktools/ai/capability/__init__.py'
text = path.read_text(encoding='utf-8')
old = 'from ._mcp import MCPCapabilityProvider, MCPRuntime, mcp_server_selector\n'
new = 'from ._mcp import MCPCapabilityProvider, MCPRuntime\n'
if text.count(old) != 1:
    raise SystemExit(f'capability public import anchor count={text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')

print('final schema fingerprint and lint closure applied')
