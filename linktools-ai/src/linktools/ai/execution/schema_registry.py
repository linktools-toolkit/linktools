"""Explicit, trusted registry for resumable output schemas."""

import hashlib
from typing import Any

from ..json import canonical_json
from ..errors import ManifestDriftError


class OutputSchemaRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[type, str]] = {}

    def register(self, schema_id: str, revision: str, model: type) -> None:
        if not schema_id or not revision:
            raise ValueError("schema_id and revision are required")
        schema = model.model_json_schema() if hasattr(model, "model_json_schema") else {}
        fingerprint = hashlib.sha256(canonical_json(schema).encode()).hexdigest()
        key = (schema_id, revision)
        current = self._entries.get(key)
        if current is not None and current[1] != fingerprint:
            raise ManifestDriftError(f"output schema fingerprint changed: {schema_id}@{revision}")
        self._entries[key] = (model, fingerprint)

    def resolve(self, schema_id: str, revision: str) -> type:
        try:
            return self._entries[(schema_id, revision)][0]
        except KeyError as exc:
            raise ManifestDriftError(f"output schema is not registered: {schema_id}@{revision}") from exc

    def fingerprint(self, schema_id: str, revision: str) -> str:
        try:
            return self._entries[(schema_id, revision)][1]
        except KeyError as exc:
            raise ManifestDriftError(f"output schema is not registered: {schema_id}@{revision}") from exc
