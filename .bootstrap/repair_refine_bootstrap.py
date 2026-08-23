#!/usr/bin/env python3
from pathlib import Path

path = Path(__file__).with_name("refine_typed_persistence_codec.py")
text = path.read_text(encoding="utf-8")

# The envelope and step decoders intentionally start from the same old snippet.
# Make only the first occurrence use str.replace(..., 1), without leaving the
# first positional `text` argument from replace_once().
marker = "normalized = _normalize_persisted_value(payload, codec)"
pos = text.find(marker)
if pos < 0:
    print("typed codec bootstrap no longer needs envelope disambiguation")
    raise SystemExit(0)
start = text.rfind("    text = ", 0, pos)
if start < 0:
    raise RuntimeError("typed envelope replacement call not found")
end = text.find("    )\n", pos)
if end < 0:
    raise RuntimeError("typed envelope replacement call end not found")
end += len("    )\n")
block = text[start:end]
old_wire = "'''    normalized = _normalize_persisted_value(payload, codec)\\n    return _decode_domain(normalized, target, codec)\\n'''"
new_wire = "'''    return _decode_domain(payload, target, codec, persisted=True)\\n'''"
if old_wire not in block or new_wire not in block:
    raise RuntimeError("typed envelope replacement block is unexpected")
fixed = (
    "    text = text.replace(\n"
    f"        {old_wire},\n"
    f"        {new_wire},\n"
    "        1,\n"
    "    )\n"
)
text = text[:start] + fixed + text[end:]
path.write_text(text, encoding="utf-8")
print("typed codec bootstrap envelope replacement repaired")
