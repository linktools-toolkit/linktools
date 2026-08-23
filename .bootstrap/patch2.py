from pathlib import Path

files = (
    'tests/ai/test_session_admission.py',
    'tests/ai/test_session_admission_conformance.py',
    'tests/ai/test_session_history_projection.py',
)
for path_value in files:
    path = Path(path_value)
    text = path.read_text(encoding='utf-8')
    count = text.count('        binding_digest="binding",\n')
    if count != 1:
        raise RuntimeError(f'{path_value}: expected one SessionRecord binding field, got {count}')
    text = text.replace(
        '        binding_digest="binding",\n',
        '        agent_digest="a" * 64,\n',
        1,
    )
    path.write_text(text, encoding='utf-8')
    print(f'{path_value}: SessionRecord now pins agent_digest')
