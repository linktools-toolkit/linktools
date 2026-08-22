#!/usr/bin/env python3
import subprocess

BLOB = "dff00f29c05c93719d7f30f36c9796e26091a800"
source = subprocess.check_output(("git", "cat-file", "blob", BLOB)).decode("ascii")
prefix = 'PAYLOAD = "'
start = source.index(prefix) + len(prefix)
end = source.index("\ndata = lzma.decompress", start)
region = source[start:end]
allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n\t ")
invalid = [(index, char) for index, char in enumerate(region) if char not in allowed]
print(f"region_length={len(region)} invalid_count={len(invalid)}")
for index, char in invalid[:40]:
    left = max(0, index - 60)
    right = min(len(region), index + 61)
    print(f"invalid index={index} char={char!r} context={region[left:right]!r}")
raise RuntimeError("payload framing diagnostic")
