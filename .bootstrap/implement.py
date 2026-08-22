#!/usr/bin/env python3
import base64
import gzip
import io
import subprocess
import tarfile
from pathlib import Path

ROOT = Path.cwd().resolve()
PARTS = sorted((ROOT / ".bootstrap").glob("source.b64.part-*"))
if not PARTS:
    raise RuntimeError("bootstrap payload parts not found")

payload = "".join(part.read_text(encoding="ascii").strip() for part in PARTS)
data = gzip.decompress(base64.b64decode(payload, validate=True))

tracked_modes: dict[str, str] = {}
for entry in subprocess.check_output(("git", "ls-files", "-s", "-z")).split(b"\0"):
    if not entry:
        continue
    metadata, raw_path = entry.split(b"\t", 1)
    tracked_modes[raw_path.decode("utf-8")] = metadata.split(b" ", 1)[0].decode("ascii")

with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
    members = archive.getmembers()
    for member in members:
        target = (ROOT / member.name).resolve()
        if target != ROOT and ROOT not in target.parents:
            raise RuntimeError(f"unsafe archive member: {member.name}")
    archive.extractall(ROOT, filter="data")

for member in members:
    path = ROOT / member.name
    if not path.is_file():
        continue
    tracked_mode = tracked_modes.get(member.name)
    current_mode = path.stat().st_mode
    if tracked_mode == "100755":
        path.chmod(current_mode | 0o111)
    elif tracked_mode == "100644":
        path.chmod(current_mode & ~0o111)
    elif member.name.endswith(".py") and path.read_bytes().startswith(b"#!"):
        path.chmod(current_mode | 0o111)
