from pathlib import Path

path = Path('linktools-ai/src/linktools/ai/runtime/state/_codec.py')
text = path.read_text(encoding='utf-8')
old = '"execution_event": "c7d10fa9a15092e7e29c503b938f12358b836ee0d578bd5faff0f0951d3bedf8"'
new = '"execution_event": "0c977a9bc5de198667dc34640acd192fe54a986227d886ffb1dea596a851c957"'
if text.count(old) != 1:
    raise SystemExit(f'execution_event fingerprint anchor count={text.count(old)}')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
print('execution_event V1 fingerprint aligned with mechanical schema descriptor')
