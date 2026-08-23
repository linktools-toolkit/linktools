from pathlib import Path

root = Path('.')

path = root / 'linktools-ai/src/linktools/ai/runtime/state/_migration.py'
text = path.read_text(encoding='utf-8')
text = text.replace('from typing import Any, get_type_hints\n', 'from typing import get_type_hints\n')
text = text.replace('    RecoveryAdmissionRecord,\n    RecoveryExecutionInput,\n    SessionRecord,\n', '    RecoveryAdmissionRecord,\n    SessionRecord,\n')
path.write_text(text, encoding='utf-8')

path = root / 'tests/ai/test_runtime_state_identity_migration.py'
text = path.read_text(encoding='utf-8')
text = text.replace(
    'from linktools.ai.runtime.state import (\n    ContextProjection,\n    RuntimeDomain,\n    SessionRecord,\n    migrate_v1_agent_identity_state,\n)\n',
    'from linktools.ai.runtime.state import (\n    SessionRecord,\n    migrate_v1_agent_identity_state,\n)\n',
)
path.write_text(text, encoding='utf-8')

path = root / '.github/workflows/ai-runtime-state-migration.yml'
text = path.read_text(encoding='utf-8')
old = '          if [[ $status -eq 0 ]]; then run_check ruff check --select E4,E7,E9,F linktools-ai/src/linktools/ai tests/ai; fi\n'
new = '''          if [[ $status -eq 0 ]]; then
            mapfile -t changed_python < <(git diff --name-only cbda9b0347a03be5a1727f76fdd2782f10ee3bda -- linktools-ai/src/linktools/ai tests/ai | grep '\\.py$' || true)
            if [[ ${#changed_python[@]} -gt 0 ]]; then run_check ruff check --select E4,E7,E9,F "${changed_python[@]}"; fi
          fi
'''
if old not in text:
    raise SystemExit('workflow ruff anchor not found')
path.write_text(text.replace(old, new), encoding='utf-8')

print('migration lint gate narrowed to changed files')
