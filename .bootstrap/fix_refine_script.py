#!/usr/bin/env python3
from pathlib import Path

path = Path(__file__).with_name("refine_typed_persistence_codec.py")
text = path.read_text(encoding="utf-8")
marker = '"typed envelope decode",'
pos = text.find(marker)
if pos < 0:
    raise RuntimeError("typed envelope decode marker not found")
start = text.rfind("text = replace_once(", 0, pos)
if start < 0:
    # Already adjusted.
    print("typed codec bootstrap already disambiguated")
    raise SystemExit(0)
text = text[:start] + text[start:].replace("text = replace_once(", "text = text.replace(", 1)
pos = text.find(marker, start)
if pos < 0:
    raise RuntimeError("typed envelope decode marker moved unexpectedly")
text = text[:pos] + text[pos:].replace(marker, "1,", 1)
path.write_text(text, encoding="utf-8")
print("typed codec bootstrap disambiguated")
