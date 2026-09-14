from pathlib import Path

path = Path("linktools-ai/src/linktools/ai/runtime/state/_commands.py")
text = path.read_text(encoding="utf-8")
old = "history = await self._promote_history_in_transaction("
if text.count(old) != 3:
    raise RuntimeError(f"expected three obsolete history assignments, found {text.count(old)}")
path.write_text(text.replace(old, "await self._promote_history_in_transaction("), encoding="utf-8")
print("obsolete timeline history assignments removed")
