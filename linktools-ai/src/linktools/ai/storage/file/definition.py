#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FileRunDefinitionStore: one JSON snapshot per run at
``{root}/{run_id}/definition.json``."""

import asyncio
import json
from datetime import datetime
from pathlib import Path

from ...run.definition import RunDefinitionSnapshot


def _validate_segment(value: str, *, kind: str) -> str:
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError(f"invalid {kind}: {value!r}")
    return value


class FileRunDefinitionStore:
    def __init__(self, *, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        d = self._root / _validate_segment(run_id, kind="run_id")
        d.mkdir(parents=True, exist_ok=True)
        return d / "definition.json"

    async def create(self, snapshot: RunDefinitionSnapshot) -> None:
        # serialized_spec + manifest are deep-frozen (mappingproxy); round-trip
        # through canonical_json to get plain JSON-native structures for dumps.
        from ...json import canonical_json
        import json as _json

        payload = {
            "run_id": snapshot.run_id,
            "runnable_type": snapshot.runnable_type,
            "runnable_id": snapshot.runnable_id,
            "serialized_spec": _json.loads(canonical_json(snapshot.serialized_spec)),
            "spec_fingerprint": snapshot.spec_fingerprint,
            "user_id": snapshot.user_id,
            "tenant_id": snapshot.tenant_id,
            "workspace": snapshot.workspace,
            "provider_revision": snapshot.provider_revision,
            "created_at": snapshot.created_at.isoformat(),
            "manifest": _json.loads(canonical_json(snapshot.manifest))
            if snapshot.manifest
            else {},
            "resumability": snapshot.resumability,
        }
        await asyncio.to_thread(
            self._path(snapshot.run_id).write_text, json.dumps(payload)
        )

    async def get(self, run_id: str) -> "RunDefinitionSnapshot | None":
        path = self._path(run_id)
        exists = await asyncio.to_thread(lambda: path.is_file())
        if not exists:
            return None
        raw = await asyncio.to_thread(path.read_text)
        data = json.loads(raw)
        return RunDefinitionSnapshot(
            run_id=data["run_id"],
            runnable_type=data["runnable_type"],
            runnable_id=data["runnable_id"],
            serialized_spec=data["serialized_spec"],
            spec_fingerprint=data["spec_fingerprint"],
            user_id=data["user_id"],
            tenant_id=data["tenant_id"],
            workspace=data["workspace"],
            provider_revision=data["provider_revision"],
            created_at=datetime.fromisoformat(data["created_at"]),
            manifest=data.get("manifest") or {},
            resumability=data.get("resumability", "resumable"),
        )
