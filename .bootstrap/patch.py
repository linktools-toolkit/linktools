from pathlib import Path

changes = {
    'tests/ai/test_session_admission_conformance.py': (
        '        binding_digest="b" * 64,\n',
        '        agent_digest="b" * 64,\n',
    ),
    'tests/ai/test_session_history_projection.py': (
        '        binding_digest="binding",\n',
        '        agent_digest="a" * 64,\n',
    ),
}
for path_value, (old, new) in changes.items():
    path = Path(path_value)
    text = path.read_text(encoding='utf-8')
    if text.count(old) != 1:
        raise RuntimeError(f'{path_value}: expected one old Session identity field, got {text.count(old)}')
    path.write_text(text.replace(old, new, 1), encoding='utf-8')
    print(f'{path_value}: SessionRecord migrated to agent_digest')
