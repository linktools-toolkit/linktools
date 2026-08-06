#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prompt snapshot freezing."""

from ...domain.prompt import PromptSnapshot
from ...foundation.digest import sha256_digest
from ...foundation.ids import deterministic_id


class PromptSnapshotService:
    def freeze(self, execution_id: str, prompt_name: str, content: bytes) -> PromptSnapshot:
        digest = sha256_digest(content)
        snapshot_id = deterministic_id(b"prompt-snapshot", execution_id, prompt_name, digest)
        return PromptSnapshot(snapshot_id=snapshot_id, prompt_name=prompt_name, digest=digest, content_ref=f"blob://prompt/{snapshot_id}")


__all__ = ["PromptSnapshotService"]
