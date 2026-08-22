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
end = source.index('"', start)
payload = "".join(source[start:end].split())
data = lzma.decompress(base64.b64decode(payload, validate=True))

with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
    members = archive.getmembers()
    for member in members:
        target = (ROOT / member.name).resolve()
        if target == ROOT or ROOT not in target.parents:
            raise RuntimeError(f"unsafe archive member: {member.name}")
    archive.extractall(ROOT, filter="data")
