#!/usr/bin/env python3
import base64
import io
import lzma
import tarfile
from pathlib import Path

ROOT = Path.cwd().resolve()
PARTS = sorted((ROOT / ".bootstrap").glob("source.b64.part-*"))
if not PARTS:
    raise RuntimeError("bootstrap payload parts not found")

payload = "".join(part.read_text(encoding="ascii").strip() for part in PARTS)
data = lzma.decompress(base64.b64decode(payload, validate=True))

with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
    for member in archive.getmembers():
        target = (ROOT / member.name).resolve()
        if target != ROOT and ROOT not in target.parents:
            raise RuntimeError(f"unsafe archive member: {member.name}")
    archive.extractall(ROOT, filter="data")
