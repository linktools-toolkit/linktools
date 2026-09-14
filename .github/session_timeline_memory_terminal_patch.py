from pathlib import Path

path = Path("linktools-ai/src/linktools/ai/runtime/_local.py")
text = path.read_text(encoding="utf-8")
old = '''            if execution.session_id is not None and status is ExecutionStatus.SUCCEEDED:\n                archive = self._step_reads[RuntimeDomain.CONVERSATION]\n                if isinstance(archive, StateStepArchive):\n                    stores.append(archive.state_store)\n        return all(\n'''
new = '''            if execution.session_id is not None and status is ExecutionStatus.SUCCEEDED:\n                archive = self._step_reads[RuntimeDomain.CONVERSATION]\n                if not isinstance(archive, StateStepArchive):\n                    return False\n                stores.append(archive.state_store)\n        return all(\n'''
count = text.count(old)
if count != 1:
    raise RuntimeError(f"expected one same-group conversation branch, found {count}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("in-memory session success routed through recovery handoff")
