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

from .container import ContainerError

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import PathType
    from .manager import ContainerManager

INDEX_SCHEMA_VERSION = 1


class ArtifactIndexError(ContainerError):
    """The Artifact Index file exists but is unusable: corrupt JSON, a
    non-object root/artifacts, or an unsupported schema_version. Distinct
    from "genuinely doesn't exist yet" (load() returns {} for that,
    unchanged) -- a caller must never treat a corrupted index as if it
    were simply empty, which would make record() silently discard every
    prior entry, and Plan/Doctor silently treat every real artifact as
    newly "added"."""


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

    def _load_raw(self) -> "dict[str, dict[str, Any]]":
        """Parse and structurally validate the index file, returning every
        field of each entry (including a legacy ``repository_url``, if
        present -- ``load()`` strips that; ``entries_with_legacy_repository_url()``
        needs to see it).

        Fail-closed: only a genuinely absent file returns ``{}``. Anything
        else wrong -- unreadable, invalid JSON, a non-object root/
        ``artifacts``/entry, or an unsupported ``schema_version`` -- raises
        ``ArtifactIndexError`` instead of silently returning ``{}``, which
        would otherwise make ``record()`` discard every prior entry and
        Plan/Doctor treat every real artifact as newly "added".
        """
        path = self.path
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except OSError as exc:
            raise ArtifactIndexError(f"cannot read artifact index {path}: {exc}") from exc
        except ValueError as exc:
            raise ArtifactIndexError(f"artifact index {path} is not valid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ArtifactIndexError(f"artifact index {path} root must be an object")
        schema_version = data.get("schema_version")
        if schema_version != INDEX_SCHEMA_VERSION:
            raise ArtifactIndexError(
                f"artifact index {path} has unsupported schema_version {schema_version!r}")
        project = data.get("project")
        if project is not None and not isinstance(project, str):
            raise ArtifactIndexError(f"artifact index {path} `project` must be a string")
        artifacts = data.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ArtifactIndexError(f"artifact index {path} `artifacts` must be an object")
        for rel_path, meta in artifacts.items():
            if not isinstance(meta, dict):
                raise ArtifactIndexError(
                    f"artifact index {path} entry {rel_path!r} must be an object")
        return artifacts

    def load(self) -> "dict[str, dict[str, Any]]":
        artifacts = self._load_raw()
        # `repository_url` (removed: could carry a Git credential) is
        # stripped on load rather than left for a caller to accidentally
        # display -- an entry still written under the old field only ever
        # existed before this fix, and gets overwritten with the new,
        # credential-free fields the next time this same artifact is
        # recorded (record() replaces an entry wholesale, not a merge).
        return {
            rel_path: {k: v for k, v in meta.items() if k != "repository_url"}
            for rel_path, meta in artifacts.items()
        }

    def entries_with_legacy_repository_url(self) -> "list[str]":
        """Relative artifact paths whose on-disk entry still carries the
        removed ``repository_url`` field -- ``load()`` already strips it
        from its result, so Doctor uses this (reading the raw file) to
        prompt a rebuild instead. Never raises -- an unusable index is
        already reported by Doctor's own ``load()`` call; this is a purely
        best-effort extra hint."""
        try:
            artifacts = self._load_raw()
        except ArtifactIndexError:
            return []
        return sorted(rel_path for rel_path, meta in artifacts.items() if "repository_url" in meta)

    def record(self, entries: "dict[str, dict[str, Any]]") -> bool:
        """Merge ``entries`` (artifact relative path -> metadata) into the
        index and write it atomically (canonical JSON, sorted, trailing
        newline). Unrelated existing entries are preserved. Returns True iff
        the on-disk index content changed.

        Holds a process-wide lock around the whole read-merge-write so two
        concurrent recordings (e.g. two containers' artifacts generated in
        the same `up`) can never lose one's entries to the other's: each
        read-then-write pair used to run unlocked, so two writers reading
        the same starting index and writing back independently would have
        the later write silently discard the earlier one's now-unseen
        entries.
        """
        with self.manager.environ.locks.process_lock("cntr:artifact-index"):
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
