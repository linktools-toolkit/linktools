#!/usr/bin/env python3
import base64
import io
import lzma
import subprocess
import tarfile
from pathlib import Path

ROOT = Path.cwd().resolve()
BLOB = "dff00f29c05c93719d7f30f36c9796e26091a800"
source = subprocess.check_output(("git", "cat-file", "blob", BLOB)).decode("ascii")
prefix = 'PAYLOAD = "'
start = source.index(prefix) + len(prefix)
code_start = source.index("data = lzma.decompress", start)
payload_region = source[start:code_start]
quote = payload_region.rfind('"')
if quote < 0:
    raise RuntimeError("bootstrap payload terminator not found")
payload = "".join(payload_region[:quote].split())
data = lzma.decompress(base64.b64decode(payload, validate=True))

with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
    members = archive.getmembers()
    for member in members:
        target = (ROOT / member.name).resolve()
        if target != ROOT and ROOT not in target.parents:
            raise RuntimeError(f"unsafe archive member: {member.name}")
    archive.extractall(ROOT, filter="data")
