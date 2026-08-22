#!/usr/bin/env python3
import base64
import gzip
import json
import subprocess
from pathlib import Path

ROOT = Path.cwd().resolve()
STAGES = (
    ROOT / ".bootstrap" / "stage1.json",
    ROOT / ".bootstrap" / "stage2.json",
)

tracked_modes: dict[str, str] = {}
for entry in subprocess.check_output(("git", "ls-files", "-s", "-z")).split(b"\0"):
    if not entry:
        continue
    metadata, raw_path = entry.split(b"\t", 1)
    tracked_modes[raw_path.decode("utf-8")] = metadata.split(b" ", 1)[0].decode("ascii")

written: set[str] = set()
for stage_path in STAGES:
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    if not isinstance(stage, dict):
        raise RuntimeError(f"invalid bootstrap stage: {stage_path.name}")
    for relative_path, encoded_content in stage.items():
        if not isinstance(relative_path, str) or not isinstance(encoded_content, str):
            raise RuntimeError(f"invalid bootstrap entry: {stage_path.name}")
        target = (ROOT / relative_path).resolve()
        if target == ROOT or ROOT not in target.parents:
            raise RuntimeError(f"unsafe bootstrap path: {relative_path}")
        content = gzip.decompress(base64.b64decode(encoded_content, validate=True))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        written.add(relative_path)

for relative_path in written:
    path = ROOT / relative_path
    tracked_mode = tracked_modes.get(relative_path)
    current_mode = path.stat().st_mode
    if tracked_mode == "100755":
        path.chmod(current_mode | 0o111)
    elif tracked_mode == "100644":
        path.chmod(current_mode & ~0o111)
    elif relative_path.endswith(".py") and path.read_bytes().startswith(b"#!"):
        path.chmod(current_mode | 0o111)
