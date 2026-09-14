from pathlib import Path

path = Path("linktools-ai/src/linktools/ai/runtime/state/_codec.py")
text = path.read_text(encoding="utf-8")
old = '''    custom_encoders = {\n        "object_ref",\n        "task_graph_view",\n'''
new = '''    custom_encoders = {\n        "object_ref",\n        "stored_user_input",\n        "task_graph_view",\n'''
if text.count(old) != 1:
    raise RuntimeError("custom encoder contract marker not found exactly once")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
