#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generated Artifact Index: ``<data_path>/generated/index.json`` records
which generated file came from which container/source and its content hash,
for Plan/Doctor to reason about drift later. Also the atomic writer every
generated-file write path (this module, the compose/Dockerfile writers)
shares.

The index never records config values, secrets, or full template context --
only a relative path, kind, owning container, sha256 and (best-effort)
source path. It never deletes stale entries/files itself; it only records.
"""
import hashlib
import json
import os
import stat
from typing import TYPE_CHECKING

from linktools import utils

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import PathType
    from .manager import ContainerManager

INDEX_SCHEMA_VERSION = 1


def atomic_write_text_if_changed(path: "PathType", content: str, encoding: str = "utf-8") -> bool:
    """Write ``content`` to ``path`` atomically. Return True iff it changed.

    ``linktools.utils.atomic_write`` replaces the target with a freshly
    created temp file (``tempfile.mkstemp``, mode 0600), which would
    otherwise silently narrow an existing file's permissions on every
    regeneration; the previous mode is restored here for an existing target.
    """
    path = str(path)
    original_mode = None
    if os.path.exists(path):
        with open(path, "r", encoding=encoding) as f:
            existing = f.read()
        if existing == content:
            return False
        original_mode = stat.S_IMODE(os.stat(path).st_mode)
    utils.atomic_write(path, content, encoding=encoding)
    if original_mode is not None:
        os.chmod(path, original_mode)
    return True


def sha256_of(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def collect_candidates(manager: "ContainerManager", containers) -> "dict[str, tuple[str, str, str]]":
    """Render each container's compose/Dockerfile candidate content
    in-memory -- never touching the real generated file on disk. Returns
    ``{absolute_destination_path: (kind, container_name, content)}``.

    Shared by ExecutionPlanner (dry-run artifact hashing) and the real
    up/restart write path, so the two can never compute a candidate's
    destination or content differently.
    """
    import yaml

    candidates: "dict[str, tuple[str, str, str]]" = {}
    for container in containers:
        compose = container.docker_compose
        if compose:
            content = yaml.safe_dump(compose, sort_keys=True, allow_unicode=False)
            dest = str(utils.join_path(manager.data_path, "compose", f"{container.name}.yml"))
            candidates[dest] = ("compose", container.name, content)
        docker_file = container.docker_file
        if docker_file:
            dest = str(utils.join_path(manager.data_path, "dockerfile", f"{container.name}.Dockerfile"))
            candidates[dest] = ("dockerfile", container.name, docker_file)
    return candidates


class ArtifactIndex:
    """Owns ``<data_path>/generated/index.json`` behind the facade."""

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    @property
    def path(self) -> str:
        return os.path.join(str(self.manager.data_path), "generated", "index.json")

    def load(self) -> "dict[str, dict[str, Any]]":
        path = self.path
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        artifacts = data.get("artifacts") if isinstance(data, dict) else None
        return artifacts if isinstance(artifacts, dict) else {}

    def record(self, entries: "dict[str, dict[str, Any]]") -> bool:
        """Merge ``entries`` (artifact relative path -> metadata) into the
        index and write it atomically (canonical JSON, sorted, trailing
        newline). Unrelated existing entries are preserved. Returns True iff
        the on-disk index content changed."""
        artifacts = self.load()
        artifacts.update(entries)
        payload = dict(
            schema_version=INDEX_SCHEMA_VERSION,
            project=self.manager.project_name,
            artifacts=artifacts,
        )
        content = json.dumps(payload, sort_keys=True, indent=2) + "\n"
        path = self.path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return atomic_write_text_if_changed(path, content)
