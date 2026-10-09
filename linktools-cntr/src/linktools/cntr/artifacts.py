#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generated Artifact Index: ``<data_path>/generated/index.json`` records
which generated file came from which container/source and its content hash,
for Plan/Doctor to reason about drift later. Also the atomic writer every
generated-file write path (this module, the compose/Dockerfile writers)
shares.

The index never records config values, secrets, or full template context --
only a relative path, kind, owning container, sha256 and (best-effort)
source path. It never scans or deletes stale files; explicit rollback may
remove entries for applied snapshots that were undone.
"""
import hashlib
import json
import os
import stat
from typing import TYPE_CHECKING

from linktools import utils

from .container import ContainerError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Callable
    from pathlib import Path
    from linktools.types import PathType
    from .container import BaseContainer
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


def docker_file_destination(container: "BaseContainer") -> "Path":
    """Return the generated Dockerfile path without rendering or writing it."""
    return utils.join_path(container.manager.data_path, "dockerfile", f"{container.name}.Dockerfile")


def compose_candidate(container: "BaseContainer") -> "tuple[Path, str] | None":
    """Serialize the Compose model without creating its destination."""
    import yaml

    compose = container.docker_compose
    if not compose:
        return None
    destination = utils.join_path(container.manager.data_path, "compose", f"{container.name}.yml")
    return destination, yaml.safe_dump(compose, sort_keys=True, allow_unicode=False)


def docker_file_candidate(container: "BaseContainer") -> "tuple[Path, str] | None":
    """Return a rendered Dockerfile and destination without writing either."""
    content = container.docker_file
    if not content:
        return None
    return docker_file_destination(container), content


def collect_candidates(manager: "ContainerManager", containers: "Iterable[BaseContainer]") -> "dict[str, tuple[str, str, str]]":
    """Collect pure Compose/Dockerfile serializations shared with file writers.

    Returns ``{absolute_destination_path: (kind, container_name, content)}``.
    The manager argument is retained for callers; each container owns its
    destination through its manager, just as it does during execution.
    """
    candidates: "dict[str, tuple[str, str, str]]" = {}
    for container in containers:
        for kind, candidate in (("compose", compose_candidate(container)),
                                ("dockerfile", docker_file_candidate(container))):
            if candidate is not None:
                destination, content = candidate
                candidates[str(destination)] = (kind, container.name, content)
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

    def record(self, entries: "dict[str, dict[str, Any]]", *,
               remove: "Iterable[str]" = ()) -> bool:
        """Merge ``entries`` and explicitly remove paths undone by rollback.
        The index is written atomically (canonical JSON, sorted, trailing
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
            for rel_path in remove:
                artifacts.pop(rel_path, None)
            payload = dict(
                schema_version=INDEX_SCHEMA_VERSION,
                project=self.manager.project_name,
                artifacts=artifacts,
            )
            content = json.dumps(payload, sort_keys=True, indent=2) + "\n"
            path = self.path
            os.makedirs(os.path.dirname(path), exist_ok=True)
            return atomic_write_text_if_changed(path, content)


class GeneratedCandidate:
    """An immutable generated tree with an atomic, reversible current link."""

    def __init__(self, container: "BaseContainer",
                 render: "Callable[[str], dict[str, str]]") -> None:
        import uuid
        self.container = container
        self.root = str(container.get_app_path("generated"))
        self.previous_id = self.current_id(self.root)
        self.generation_id = self.previous_id or uuid.uuid4().hex
        files = render(self.generation_id)
        self.changed = not self.previous_id or not self.matches(files)
        if self.changed and self.previous_id:
            self.generation_id = uuid.uuid4().hex
            files = render(self.generation_id)
        self.path = os.path.join(self.root, self.generation_id)
        self.changed_files = tuple(sorted(name for name, content in files.items()
                                          if self.read_previous(name) != content))
        if self.changed:
            os.makedirs(self.path, mode=0o755)
            entries = {}
            for name, content in files.items():
                if os.path.isabs(name) or '..' in name.split('/'):
                    raise ContainerError("Generated file must stay within its candidate tree")
                path = os.path.join(self.path, name)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                atomic_write_text_if_changed(path, content)
                entries[os.path.relpath(path, str(container.manager.data_path))] = dict(
                    kind="generated-config", container=container.name, sha256=sha256_of(content))
            container.manager.artifact_index.record(entries)

    @classmethod
    def current_id(cls, root: str) -> "str | None":
        link = os.path.join(root, "current")
        if not os.path.lexists(link):
            return None
        if not os.path.islink(link):
            raise ContainerError("Generated current must be a symbolic link")
        value = os.readlink(link)
        if not value or os.path.basename(value) != value or value in (".", ".."):
            raise ContainerError("Invalid generated current target")
        return value

    def read_previous(self, name: str) -> "str | None":
        if self.previous_id:
            try:
                with open(os.path.join(self.root, self.previous_id, name), encoding="utf-8") as stream:
                    return stream.read()
            except FileNotFoundError:
                pass
        return None

    def matches(self, files: "dict[str, str]") -> bool:
        root = os.path.join(self.root, self.generation_id)
        actual = set()
        for directory, _, names in os.walk(root):
            actual.update(os.path.relpath(os.path.join(directory, name), root) for name in names)
        return actual == set(files) and all(self.read_previous(name) == text for name, text in files.items())

    def activate(self, generation_id: "str | None") -> None:
        import uuid
        current = os.path.join(self.root, "current")
        if generation_id is None:
            if os.path.lexists(current):
                os.unlink(current)
            return
        temporary = os.path.join(self.root, ".current-" + uuid.uuid4().hex)
        os.symlink(generation_id, temporary)
        try:
            os.replace(temporary, current)
        finally:
            if os.path.lexists(temporary):
                os.unlink(temporary)

    def publish(self) -> None:
        self.activate(self.generation_id)

    def restore(self) -> None:
        self.activate(self.previous_id)


class AppliedServiceModels:
    """Track each service's applied model, retaining project support for rollback."""

    def __init__(self, manager: "ContainerManager", model: dict) -> None:
        from types import MappingProxyType
        import yaml

        self.manager = manager
        self.root = os.path.join(str(manager.data_path), "compose", "applied", "services")
        if not isinstance(model, dict) or not isinstance(model.get("services"), dict):
            raise ContainerError("Resolved Compose model must contain a services mapping")
        resolved = self._normalize(model)
        current, previous = {}, {}
        changed = set(model["services"])
        for service, spec in model["services"].items():
            if not isinstance(service, str) or not service or not isinstance(spec, dict):
                raise ContainerError("Resolved Compose services must have names and mapping definitions")
            # Compose validates dependencies even with --no-deps. Each service
            # snapshot therefore retains its full project's rollback support.
            current[service] = resolved
            path = self._path(service)
            try:
                with open(path, encoding="utf-8") as stream:
                    saved = yaml.safe_load(stream)
            except FileNotFoundError:
                if os.path.lexists(path):
                    raise ContainerError("Cannot read applied Compose model for service {}".format(service)) from None
                continue
            except (OSError, UnicodeError, yaml.YAMLError):
                raise ContainerError("Cannot read applied Compose model for service {}".format(service)) from None
            if (not isinstance(saved, dict) or not isinstance(saved.get("services"), dict)
                    or service not in saved["services"]
                    or any(not isinstance(name, str) or not name or not isinstance(definition, dict)
                           for name, definition in saved["services"].items())):
                raise ContainerError("Invalid applied Compose model for service {}".format(service))
            previous[service] = self._normalize(saved)
            if self._projection(saved, service) == self._projection(model, service):
                changed.remove(service)
        self.current = MappingProxyType(current)
        self.previous = MappingProxyType(previous)
        self.changed_services = frozenset(changed)

    @classmethod
    def _projection(cls, model: dict, service: str) -> str:
        shared = {key: value for key, value in model.items() if key != "services"}
        return cls._normalize(dict(shared, services={service: model["services"][service]}))

    @classmethod
    def _normalize(cls, model: dict) -> str:
        import yaml
        try:
            # Compose's resolved model is JSON-compatible. Round-tripping also
            # removes YAML aliases whose spelling depends on object identity.
            value = json.loads(json.dumps(model, sort_keys=True, allow_nan=False))
            return yaml.safe_dump(value, sort_keys=True, allow_unicode=False)
        except (TypeError, ValueError, yaml.YAMLError):
            raise ContainerError("Applied Compose model is not a valid resolved model") from None

    def _path(self, service: str) -> str:
        return os.path.join(self.root, service.encode("utf-8").hex() + ".yml")

    def record(self, services: "Iterable[str]") -> None:
        selected = tuple(dict.fromkeys(services))
        if any(service not in self.current for service in selected):
            raise ContainerError("Cannot record services absent from the resolved Compose model")
        if not selected:
            return
        os.makedirs(self.root, mode=0o700, exist_ok=True)
        os.chmod(os.path.dirname(self.root), 0o700)
        os.chmod(self.root, 0o700)
        entries = {}
        for service in selected:
            content = self.current[service]
            path = self._path(service)
            # Resolved environments may contain secrets; the atomic writer's
            # fresh 0600 file must not inherit a permissive existing mode.
            utils.atomic_write(path, content, encoding="utf-8")
            entries[os.path.relpath(path, str(self.manager.data_path))] = dict(
                kind="compose-applied-service", container=service, sha256=sha256_of(content))
        self.manager.artifact_index.record(entries)

    def restore(self, services: "Iterable[str]") -> None:
        """Restore snapshot identities after their runtime rollback succeeds."""
        entries = {}
        removed = []
        for service in dict.fromkeys(services):
            path = self._path(service)
            previous = self.previous.get(service)
            if previous is None:
                if os.path.exists(path):
                    os.unlink(path)
                removed.append(os.path.relpath(path, str(self.manager.data_path)))
                continue
            utils.atomic_write(path, previous, encoding="utf-8")
            entries[os.path.relpath(path, str(self.manager.data_path))] = dict(
                kind="compose-applied-service", container=service, sha256=sha256_of(previous))
        if entries or removed:
            self.manager.artifact_index.record(entries, remove=removed)
