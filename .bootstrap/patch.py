from pathlib import Path

path = Path('linktools-ai/src/linktools/ai/runtime/state/_contracts.py')
text = path.read_text(encoding='utf-8')
old = '''    created_at: datetime\n    updated_at: datetime\n    memory_scope: str | None = None\n    conversation_step_run_id: str | None = None\n    result: ResultRecord | None = None\n    planning: bool\n    thinking: bool\n    binding: AgentBindingSnapshot\n'''
new = '''    created_at: datetime\n    updated_at: datetime\n    planning: bool\n    thinking: bool\n    binding: AgentBindingSnapshot\n    memory_scope: str | None = None\n    conversation_step_run_id: str | None = None\n    result: ResultRecord | None = None\n'''
if text.count(old) != 1:
    raise RuntimeError('ExecutionRecord field order pattern mismatch')
path.write_text(text.replace(old, new, 1), encoding='utf-8')
print('mandatory ExecutionRecord binding fields reordered before defaults')
