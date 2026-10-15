#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded, explicit email-body conversion for the email-preview Skill."""

import argparse
import json
from email import policy
from email.parser import BytesParser
from pathlib import Path

_MAX_BYTES = 1024 * 1024


def extract(source: Path, target: Path) -> None:
    with source.open("rb") as stream:
        body = stream.read(_MAX_BYTES + 1)
    if len(body) > _MAX_BYTES:
        raise ValueError("email exceeds the 1 MiB input budget")
    message = BytesParser(policy=policy.default).parsebytes(body)
    if message.defects or not any(message.get(name) for name in ("From", "Subject", "To")):
        raise ValueError("email is malformed or lacks message headers")
    selected = None
    selected_index = None
    for index, part in enumerate(message.walk()):
        if part.defects:
            raise ValueError("email contains malformed MIME content")
        if part.get_content_type() == "text/plain" and part.get_content_disposition() != "attachment":
            selected, selected_index = part, index
            break
    if selected is None:
        raise ValueError("email has no plain-text body; explicit HTML conversion is required")
    raw = selected.get_payload(decode=True)
    if not isinstance(raw, bytes):
        raise ValueError("email body cannot be decoded")
    charset = selected.get_content_charset() or "ascii"
    text = raw.decode(charset, errors="strict")
    encoded = text.encode("utf-8")
    if len(encoded) > _MAX_BYTES:
        raise ValueError("decoded body exceeds the 1 MiB output budget")
    provenance = {
        "source": str(source), "mime_part": selected_index, "charset": charset,
        "lossy": True,
        "omitted": ["headers", "other MIME parts", "attachments", "HTML layout"],
    }
    metadata = target.with_name(target.name + ".source.json")
    if target.exists() or metadata.exists():
        raise FileExistsError("output already exists")
    with target.open("xb") as stream:
        stream.write(encoded)
    with metadata.open("x", encoding="utf-8") as stream:
        json.dump(provenance, stream, ensure_ascii=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    extract(args.source, args.target)


if __name__ == "__main__":
    main()
