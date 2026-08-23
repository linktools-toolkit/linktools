from pathlib import Path

path = Path('tests/ai/test_agent_composition.py')
text = path.read_text(encoding='utf-8')
old = 'from linktools.ai.agent._output import bind_output\n'
if text.count(old) != 1:
    raise RuntimeError('bind_output import pattern mismatch')
path.write_text(text.replace(old, '', 1), encoding='utf-8')
print('stale agent composition import removed')
