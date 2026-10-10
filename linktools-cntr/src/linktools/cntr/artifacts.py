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

from .errors import ContainerError

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, Callable
    from pathlib import Path
    from linktools.types import PathType
    from .container import BaseContainer
    from .context import OperationContext
    from typing import Mapping
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


def atomic_write_text_if_changed(path: "PathType", content: str, encoding: str = "utf-8", *,
                                 mode: "int | None" = None) -> bool:
    """Write atomically; return whether content changed, not permissions.

    Without ``mode``, preserve existing permissions. An explicit mode also
    applies when content is unchanged. The replacement starts private so
    sensitive output never inherits a previously permissive mode.
    """
    path = str(path)
    original_mode = None
    if os.path.exists(path):
        original_mode = stat.S_IMODE(os.stat(path).st_mode)
        with open(path, "r", encoding=encoding) as f:
            existing = f.read()
        if existing == content:
            if mode is not None and original_mode != mode:
                os.chmod(path, mode)
            return False
    utils.atomic_write(path, content, encoding=encoding)
    target_mode = original_mode if mode is None else mode
    if target_mode is not None:
        os.chmod(path, target_mode)
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


def stage_files(container: "BaseContainer", files: "Mapping[str, str]", *,
                mode: int = 0o600, group: "int | None" = None) -> "Path":
    """Materialize immutable inputs; no current pointer or service is changed."""
    import shutil
    import tempfile
    from pathlib import Path, PurePosixPath

    for name in files:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ContainerError("Prepared file must stay within its tree: " + name)
    payload = json.dumps([mode, group, sorted(files.items())], ensure_ascii=True,
                         separators=(",", ":"))
    root = container.get_app_path("generated")
    destination = root / sha256_of(payload)
    if destination.exists():
        actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*")
                  if path.is_file()}
        if actual != set(files) or any(
                (destination / name).is_symlink() or
                (destination / name).read_text(encoding="utf-8") != content or
                stat.S_IMODE((destination / name).stat().st_mode) != mode or
                (group is not None and (destination / name).stat().st_gid != group)
                for name, content in files.items()):
            raise ContainerError("Prepared file tree was modified: " + str(destination))
        return destination
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=str(root)))
    try:
        container.runtime.chmod(temporary, 0o755)
        for name, content in files.items():
            path = temporary / name
            path.parent.mkdir(parents=True, exist_ok=True)
            utils.atomic_write(path, content, encoding="utf-8")
            if group is not None and path.stat().st_gid != group:
                container.runtime.create_process(
                    "chgrp", str(group), str(path), privilege=True).check_call()
            container.runtime.chmod(path, mode)
        os.rename(str(temporary), str(destination))
    except BaseException:
        shutil.rmtree(str(temporary))
        raise
    producers = sorted({
        name for name, declarations in getattr(container.manager, "integration_snapshot", {}).items()
        for declaration in declarations
        if declaration.consumer == container.name or
        getattr(getattr(declaration, "expose", None), "consumer", None) == container.name
    })
    container.manager.artifact_index.record({
        os.path.relpath(str(destination / name), str(container.manager.data_path)): {
            "kind": "generated-config", "container": container.name,
            "sha256": sha256_of(content), "producers": producers,
        } for name, content in files.items()
    })
    return destination


def bind_prepared_files(context: "OperationContext", model: dict,
                        previous: "Mapping[str, str]") -> dict:
    """Bind immutable inputs, reusing unchanged single-file mounts per consumer."""
    import yaml
    from pathlib import Path

    roots = [(container.get_app_path("generated"), context.prepared_files[container.name])
             for container in context.containers if container.name in context.prepared_files]
    services = dict(model["services"])
    for service, spec in model["services"].items():
        old = yaml.safe_load(previous[service])["services"][service] if service in previous else {}
        old_mounts = {item["target"]: item for item in old.get("volumes", ())
                      if isinstance(item, dict) and item.get("type") == "bind"}
        volumes = []
        changed = False
        for item in spec.get("volumes", ()):
            replacement = item
            if isinstance(item, dict) and item.get("type") == "bind":
                source = Path(item["source"])
                for root, candidate in roots:
                    try:
                        relative = source.relative_to(root / "current")
                    except ValueError:
                        continue
                    prepared = candidate / relative
                    if not prepared.exists():
                        raise ContainerError("Missing prepared input for {}: {}".format(service, relative))
                    prior = old_mounts.get(item["target"])
                    if prior is not None and prepared.is_file():
                        old_source = Path(prior["source"])
                        try:
                            old_relative = old_source.relative_to(root)
                        except ValueError:
                            old_relative = None
                        if (old_relative is not None and old_relative.parts
                                and old_relative.parts[0] != "current" and old_source.is_file()
                                and old_source.stat().st_mode == prepared.stat().st_mode
                                and old_source.stat().st_gid == prepared.stat().st_gid
                                and old_source.read_bytes() == prepared.read_bytes()):
                            prepared = old_source
                    replacement = dict(item, source=str(prepared))
                    changed = True
                    break
            volumes.append(replacement)
        if changed:
            services[service] = dict(spec, volumes=volumes)
    return dict(model, services=services)


def publish_prepared_files(context: "OperationContext", services: "Iterable[str]") -> None:
    """Expose confirmed inputs for later read-only Compose rendering.

    Running containers use immutable mount sources, never the current symlink.
    AppliedServiceModels remains the authority for each running service.
    """
    import uuid
    import yaml
    from pathlib import Path

    services = tuple(services)
    unapplied = set(context.initial_running_services) - set(services)
    legacy_sources = [Path(item["source"]) for service in unapplied
                      if service in context.service_models.previous
                      for item in yaml.safe_load(context.service_models.previous[service])["services"][service].get("volumes", ())
                      if isinstance(item, dict) and item.get("type") == "bind"]
    sources = [Path(item["source"]) for service in services
               for item in context.compose_model["services"][service].get("volumes", ())
               if isinstance(item, dict) and item.get("type") == "bind"]
    for candidate in context.prepared_files.values():
        used = False
        for source in sources:
            try:
                source.relative_to(candidate)
            except ValueError:
                continue
            used = True
            break
        if not used:
            continue
        current = candidate.parent / "current"
        # An unselected legacy service may still dereference generated/current.
        # Its existing input must not change as a side effect of this deployment.
        if any(source == candidate.parent or source == current or
               current in source.parents for source in legacy_sources):
            continue
        if current.is_symlink() and os.readlink(str(current)) == candidate.name:
            continue
        if os.path.lexists(str(current)) and not current.is_symlink():
            raise ContainerError("Generated current must be a symbolic link")
        temporary = candidate.parent / (".current-" + uuid.uuid4().hex)
        temporary.symlink_to(candidate.name)
        try:
            os.replace(str(temporary), str(current))
        finally:
            if temporary.is_symlink():
                temporary.unlink()


def prune_prepared_files(context: "OperationContext", models: "AppliedServiceModels") -> None:
    """Retain every applied input and the preceding rollback inputs."""
    import shutil
    import yaml
    from pathlib import Path

    references = []
    for collection in (models.current, models.previous):
        for service, text in collection.items():
            spec = yaml.safe_load(text)["services"][service]
            references.extend(Path(item["source"]) for item in spec.get("volumes", ())
                              if isinstance(item, dict) and item.get("type") == "bind")
    for candidate in context.prepared_files.values():
        root = candidate.parent
        if any(source == root or source == root / "current" or
               root / "current" in source.parents for source in references):
            # A legacy directory model does not identify its concrete file tree.
            # Keep those inputs until subsequent applied models use immutable paths.
            continue
        keep = {candidate.name}
        current = root / "current"
        if current.is_symlink():
            keep.add(os.readlink(str(current)))
        for source in references:
            try:
                relative = source.relative_to(root)
            except ValueError:
                continue
            if relative.parts:
                keep.add(relative.parts[0])
        for path in root.iterdir():
            if (path.name in keep or len(path.name) not in (32, 64)
                    or any(char not in "0123456789abcdef" for char in path.name)
                    or path.is_symlink() or not path.is_dir()):
                continue
            index = models.manager.artifact_index
            prefix = os.path.relpath(str(path), str(models.manager.data_path)) + os.sep
            index.record({}, remove=tuple(key for key in index.load() if key.startswith(prefix)))
            shutil.rmtree(str(path))
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
        spec = model["services"][service]
        shared = {key: value for key, value in model.items()
                  if key not in ("services", "networks", "volumes", "secrets", "configs")}
        for category in ("networks", "volumes", "secrets", "configs"):
            definitions = model.get(category, {})
            if category == "networks":
                if spec.get("network_mode") or spec.get("networks") == []:
                    names = ()
                else:
                    names = spec.get("networks") or ("default",)
            elif category == "volumes":
                names = (item.get("source") for item in spec.get("volumes", ())
                         if isinstance(item, dict) and item.get("type") == "volume")
            else:
                names = (item if isinstance(item, str) else item.get("source")
                         for item in spec.get(category, ()))
            shared[category] = {name: definitions[name] for name in names if name in definitions}
        return cls._normalize(dict(shared, services={service: spec}))

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


    def set_model(self, model: dict) -> None:
        """Compare prepared mount identities with the same captured old models."""
        from types import MappingProxyType
        import yaml
        normalized = self._normalize(model)
        self.current = MappingProxyType({name: normalized for name in model["services"]})
        self.changed_services = frozenset(
            name for name in model["services"] if name not in self.previous or
            self._projection(yaml.safe_load(self.previous[name]), name) != self._projection(model, name))