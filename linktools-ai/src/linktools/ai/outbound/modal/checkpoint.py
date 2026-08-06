#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Safe workspace checkpoint adapter."""

import hashlib
import json
from pathlib import PurePosixPath

class WorkspaceCheckpointAdapter:
    def __init__(self, store: object) -> None:
        self._store = store

    async def capture(self, manifest: bytes) -> str:
        self._validate_manifest(manifest)
        digest = hashlib.sha256(manifest).hexdigest()
        await self._store.put(f"checkpoints/{digest}", manifest)
        return digest

    async def restore(self, checkpoint_id: str) -> bytes:
        if len(checkpoint_id) != 64 or any(char not in "0123456789abcdef" for char in checkpoint_id):
            raise ValueError("invalid checkpoint digest")
        manifest = await self._store.get(f"checkpoints/{checkpoint_id}")
        if hashlib.sha256(manifest).hexdigest() != checkpoint_id:
            raise ValueError("checkpoint digest mismatch")
        self._validate_manifest(manifest)
        return manifest

    @staticmethod
    def _validate_manifest(manifest: bytes) -> None:
        try:
            value = json.loads(manifest)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("checkpoint manifest must be JSON") from error
        files = value.get("files") if isinstance(value, dict) else None
        if not isinstance(files, list) or len(files) > 100_000:
            raise ValueError("checkpoint manifest has an invalid file list")
        total_size = 0
        for item in files:
            if not isinstance(item, dict) or item.get("kind", "file") != "file":
                raise ValueError("checkpoint contains an unsupported filesystem entry")
            path = item.get("path")
            size = item.get("size")
            digest = item.get("digest")
            if not isinstance(path, str) or "\x00" in path:
                raise ValueError("checkpoint path is invalid")
            normalized = PurePosixPath(path)
            if normalized.is_absolute() or ".." in normalized.parts or str(normalized) != path:
                raise ValueError("checkpoint path is not normalized")
            if not isinstance(size, int) or size < 0 or size > 500 * 1024 * 1024:
                raise ValueError("checkpoint file size is invalid")
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError("checkpoint file digest is invalid")
            total_size += size
            if total_size > 5 * 1024 * 1024 * 1024:
                raise ValueError("checkpoint workspace is too large")


__all__ = ["WorkspaceCheckpointAdapter"]
