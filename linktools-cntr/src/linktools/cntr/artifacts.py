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
CONFIG_REVISION_LABEL = "io.linktools.cntr.config-revision"


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
    directories = {"."}
    for name in files:
        directories.update(str(parent) for parent in PurePosixPath(name).parents)
    directory_mode = 0o755
    payload = json.dumps([mode, group, directory_mode, sorted(files.items())], ensure_ascii=True,
                         separators=(",", ":"))
    root = container.get_app_path("generated")
    destination = root / sha256_of(payload)
    if destination.is_symlink():
        raise ContainerError("Prepared file tree was modified: " + str(destination))
    if destination.exists():
        entries = tuple(destination.rglob("*"))
        actual = {path.relative_to(destination).as_posix() for path in entries}
        expected = set(files) | (directories - {"."})
        if (actual != expected or
                any(path.is_symlink() for path in entries) or any(
                    not (destination / name).is_dir() or
                    stat.S_IMODE((destination / name).stat().st_mode) != directory_mode
                    for name in directories)):
            raise ContainerError("Prepared file tree was modified: " + str(destination))
        if any(
                not (destination / name).is_file() or
                (destination / name).read_text(encoding="utf-8") != content or
                stat.S_IMODE((destination / name).stat().st_mode) != mode or
                (group is not None and (destination / name).stat().st_gid != group)
                for name, content in files.items()):
            raise ContainerError("Prepared file tree was modified: " + str(destination))
    else:
        root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".prepare-", dir=str(root)))
        try:
            for name in sorted(directories, key=lambda value: len(PurePosixPath(value).parts)):
                directory = temporary / name
                directory.mkdir(parents=True, exist_ok=True)
                os.chmod(directory, directory_mode)
            for name, content in files.items():
                path = temporary / name
                path.parent.mkdir(parents=True, exist_ok=True)
                utils.atomic_write(path, content, encoding="utf-8")
                if group is not None and path.stat().st_gid != group:
                    unprivileged = (os.name == "posix" and path.stat().st_uid == os.geteuid() and
                                    group in {os.getegid(), *os.getgroups()})
                    container.runtime.create_process(
                        "chgrp", str(group), str(path), privilege=not unprivileged).check_call()
                container.runtime.chmod(path, mode)
            os.rename(str(temporary), str(destination))
        except BaseException:
            shutil.rmtree(str(temporary))
            raise
    producers = sorted({
        name for name, declarations in getattr(container.manager, "integration_snapshot", {}).items()
        for declaration in declarations
        if declaration.consumer == container.name
    })
    container.manager.artifact_index.record({
        os.path.relpath(str(destination / name), str(container.manager.data_path)): {
            "kind": "generated-config", "container": container.name,
            "sha256": sha256_of(content), "producers": producers,
        } for name, content in files.items()
    })
    return destination


def apply_config_revisions(context: "OperationContext", model: dict,
                           services: "Iterable[str] | None" = None) -> dict:
    """Use declared configuration identity without changing any Compose mount."""
    from copy import deepcopy

    result = deepcopy(model)
    selected = context.target_services if services is None else tuple(services)
    if selected is None:
        selected = tuple(model["services"])
    declared = set()
    for service in selected:
        revisions = set()
        for container in context.project_containers:
            if service not in container.services:
                continue
            revision = container.get_config_revision(service)
            if revision is None:
                continue
            if not isinstance(revision, str) or not revision:
                raise ContainerError("Configuration revision must be a nonempty string for service " + service)
            prepared = context.prepared_dirs.get(container.name)
            if prepared is not None and revision != prepared.name:
                raise ContainerError("Configuration revision must identify the prepared tree for " + container.name)
            declared.add(container.name)
            revisions.add(revision)
        if not revisions:
            continue
        if len(revisions) != 1:
            raise ContainerError("Conflicting configuration revisions for shared service " + service)
        revision = next(iter(revisions))
        result["services"][service].setdefault("labels", {})[CONFIG_REVISION_LABEL] = revision
    for container in context.project_containers:
        if (container.name in context.prepared_dirs and set(selected).intersection(container.services)
                and container.name not in declared):
            raise ContainerError("Generated configuration consumer {} must declare get_config_revision; "
                                 "update its container definition before starting it".format(container.name))
    return result


def _service_support(model: dict, service: str) -> dict:
    spec = model["services"][service]
    support = {key: value for key, value in model.items()
               if key not in ("services", "networks", "volumes", "secrets", "configs")}
    for category in ("networks", "volumes", "secrets", "configs"):
        definitions = model.get(category, {})
        if category == "networks":
            names = () if spec.get("network_mode") or spec.get("networks") == [] else (
                spec.get("networks") or ("default",))
        elif category == "volumes":
            names = (item.get("source") for item in spec.get("volumes", ())
                     if isinstance(item, dict) and item.get("type") == "volume")
        else:
            names = (item if isinstance(item, str) else item.get("source")
                     for item in spec.get(category, ()))
        support[category] = {name: definitions[name] for name in names if name in definitions}
    return support


def bind_consumer_revisions(context: "OperationContext", model: dict,
                            models: "AppliedServiceModels", services: "Iterable[str]") -> dict:
    """Update only a consumer's revision while retaining its applied configuration."""
    from copy import deepcopy

    result = deepcopy(model)
    services = tuple(dict.fromkeys(services))
    categories = ("networks", "volumes", "secrets", "configs")
    protected = {category: {} for category in categories}
    targets = context.target_services if context.target_services is not None else model["services"]
    for service in targets:
        if service in services or service not in model["services"]:
            continue
        support = _service_support(model, service)
        for category in categories:
            protected[category].update(support[category])
    for service in services:
        previous = models.previous_model(service)
        if previous is None or service not in previous["services"]:
            raise ContainerError("No applied Compose model for configuration consumer {}; "
                                 "explicitly select its container".format(service))
        if not previous["services"][service].get("labels", {}).get(CONFIG_REVISION_LABEL):
            raise ContainerError("Configuration consumer {} has no applied configuration revision; "
                                 "explicitly select its container to migrate".format(service))
        for category, value in _service_support(previous, service).items():
            if category in ("networks", "volumes", "secrets", "configs"):
                if not value:
                    continue
                target = result.setdefault(category, {})
                for name, definition in value.items():
                    if name in protected[category] and protected[category][name] != definition:
                        raise ContainerError("Applied {} {!r} conflicts for configuration consumer {}; "
                                             "explicitly select its container".format(category, name, service))
                    target[name] = definition
                    protected[category][name] = definition
            elif category in result and result[category] != value:
                raise ContainerError("Applied Compose {!r} conflicts for configuration consumer {}; "
                                     "explicitly select its container".format(category, service))
            else:
                result[category] = value
        spec = previous["services"][service]
        result["services"][service] = spec
    return apply_config_revisions(context, result, services)


def _unapplied_running_services(context: "OperationContext", services: "Iterable[str]") -> "set[str]":
    unapplied = set(context.initial_runtime_state.running_services) - set(services)
    if context.is_full_project:
        declared = {service for container in context.project_containers for service in container.services}
        unapplied.intersection_update(declared)
    return unapplied


def _switch_current(root: "Path", revision: "str | None") -> None:
    import uuid

    current = root / "current"
    if revision is None:
        if current.is_symlink():
            current.unlink()
        return
    if current.is_symlink() and os.readlink(str(current)) == revision:
        return
    temporary = root / (".current-" + uuid.uuid4().hex)
    temporary.symlink_to(revision)
    try:
        os.replace(str(temporary), str(current))
    finally:
        if temporary.is_symlink():
            temporary.unlink()


def switch_config_revisions(context: "OperationContext",
                            revisions: "Mapping[str, str | None]") -> "dict[str, str | None]":
    """Switch consumer pointers after validation, returning their prior revisions."""
    from pathlib import Path

    containers = {container.name: container for container in context.project_containers}
    roots, previous = {}, {}
    for name, revision in revisions.items():
        if name not in containers:
            raise ContainerError("Unknown configuration consumer: " + name)
        root = containers[name].get_app_path("generated")
        current = root / "current"
        if os.path.lexists(str(current)) and not current.is_symlink():
            raise ContainerError("Generated current must be a symbolic link: " + name)
        previous[name] = os.readlink(str(current)) if current.is_symlink() else None
        for value in (revision, previous[name]):
            if value is not None and (not isinstance(value, str) or not value or
                                      value in (".", "..", "current") or Path(value).name != value or
                                      "/" in value or "\\" in value):
                raise ContainerError("Invalid generated configuration revision for " + name)
        if revision is not None and (not (root / revision).is_dir() or (root / revision).is_symlink()):
            raise ContainerError("Missing generated configuration revision for " + name)
        roots[name] = root
    changed = []
    try:
        for name, revision in revisions.items():
            if revision != previous[name]:
                _switch_current(roots[name], revision)
                changed.append(name)
    except BaseException as error:
        for name in reversed(changed):
            try:
                _switch_current(roots[name], previous[name])
            except OSError as recovery_error:
                raise ContainerError("Configuration pointer switch failed: {}; restoring {} failed: {}".format(
                    error, name, recovery_error)) from error
        raise
    return previous


def prune_prepared_files(context: "OperationContext", models: "AppliedServiceModels") -> None:
    """Retain successful and rollback revisions, including known legacy inputs."""
    import shutil
    unapplied = _unapplied_running_services(context, context.target_services or ())
    if models.untracked_services.intersection(unapplied):
        return
    specifications = []
    for collection, read in ((models.current, models.current_model), (models.previous, models.previous_model)):
        for service in collection:
            specifications.append((service, read(service)["services"][service]))
    containers = {container.name: container for container in context.project_containers}
    declared = {service for container in containers.values() for service in container.services}
    for name, candidate in context.prepared_dirs.items():
        owner = containers[name]
        configured = {service for service in owner.services if owner.get_config_revision(service) is not None}
        root = candidate.parent
        keep = {candidate.name}
        current = root / "current"
        if current.is_symlink():
            keep.add(os.readlink(str(current)))
        unknown = False
        for service, spec in specifications:
            if service not in configured and not (service not in declared and service in unapplied):
                continue
            revision = spec.get("labels", {}).get(CONFIG_REVISION_LABEL)
            if revision:
                keep.add(revision)
            else:
                unknown = True
        if unknown:
            continue
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

    def __init__(self, manager: "ContainerManager", model: dict, *,
                 retained_services: "Iterable[str]" = ()) -> None:
        from types import MappingProxyType
        import yaml

        self.manager = manager
        self.root = os.path.join(str(manager.data_path), "compose", "applied", "services")
        if not isinstance(model, dict) or not isinstance(model.get("services"), dict):
            raise ContainerError("Resolved Compose model must contain a services mapping")
        previous = {}
        loaded = {}
        self._parsed_models = {}
        for service, spec in model["services"].items():
            if not isinstance(service, str) or not service or not isinstance(spec, dict):
                raise ContainerError("Resolved Compose services must have names and mapping definitions")
        retained_services = tuple(retained_services)
        for service in dict.fromkeys(tuple(model["services"]) + retained_services):
            if not isinstance(service, str) or not service:
                raise ContainerError("Retained Compose services must have names")
            path = self._path(service)
            try:
                with open(path, encoding="utf-8") as stream:
                    text = stream.read()
                if text not in loaded:
                    saved = yaml.safe_load(text)
                    loaded[text] = saved, self._normalize(saved)
                saved, normalized = loaded[text]
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
            previous[service] = normalized
            if normalized not in self._parsed_models:
                self._parsed_models[normalized] = saved if text == normalized else yaml.safe_load(normalized)
        self.previous = MappingProxyType(previous)
        self.set_model(model)
        self.untracked_services = frozenset(service for service in retained_services if service not in previous)

    def retain_previous(self, models: "Mapping[str, str]") -> None:
        """Capture resolved legacy inputs without replacing per-service snapshots."""
        from types import MappingProxyType
        previous = dict(models)
        previous.update(self.previous)
        self.previous = MappingProxyType(previous)
        self.untracked_services = self.untracked_services.difference(previous)

    def previous_model(self, service: str) -> "dict | None":
        """Decode a separate working copy without exposing mutable snapshot state."""
        from copy import deepcopy
        text = self.previous.get(service)
        return deepcopy(self._decode_model(text)) if text is not None else None

    def current_model(self, service: str) -> "dict | None":
        """Decode a separate working copy of the current service snapshot."""
        from copy import deepcopy
        text = self.current.get(service)
        return deepcopy(self._decode_model(text)) if text is not None else None

    def _decode_model(self, text: str) -> dict:
        import yaml
        if text not in self._parsed_models:
            self._parsed_models[text] = yaml.safe_load(text)
        return self._parsed_models[text]

    @classmethod
    def _projection(cls, model: dict, service: str) -> str:
        spec = {key: value for key, value in model["services"][service].items() if key != "build"}
        return cls._normalize(dict(_service_support(model, service), services={service: spec}))

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
        normalized = self._normalize(model)
        # Compose validates dependencies even with --no-deps, so each service
        # snapshot retains the full project's support declarations.
        self.current = MappingProxyType({name: normalized for name in model["services"]})
        self.changed_services = frozenset(
            name for name in model["services"] if name not in self.previous or
            self._projection(self._decode_model(self.previous[name]), name) != self._projection(model, name))
