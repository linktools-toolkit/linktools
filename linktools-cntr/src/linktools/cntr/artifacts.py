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
    from typing import Any
    from pathlib import Path
    from linktools.types import PathType
    from .container import BaseContainer
    from .context import OperationContext
    from typing import Mapping
    from .manager import ContainerManager
    from .lifecycle.hooks import Hook

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


def prepared_files_path(root: "PathType", files: "Mapping[str, str]", *,
                        revision: "str | None" = None, mode: int = 0o600,
                        group: "int | None" = None) -> "Path":
    """Locate an immutable tree without writing or depending on a container."""
    from pathlib import Path, PurePosixPath

    for name in files:
        path = PurePosixPath(name)
        if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ContainerError("Prepared file must stay within its tree: " + name)
    if revision is not None:
        if not isinstance(revision, str):
            raise ContainerError("Prepared file revision must be a string or None")
        return Path(root) / ("revision-" + sha256_of(revision))
    payload = json.dumps([mode, group, 0o755, sorted(files.items())], ensure_ascii=True,
                         separators=(",", ":"))
    return Path(root) / sha256_of(payload)


def _is_prepared_hook(hook: "Hook") -> bool:
    """Identify a generated-file hook by its exact registered provenance."""
    from .lifecycle.hooks import HookPhase

    path = hook.metadata.get("path")
    return (hook.phase == HookPhase.BEFORE_START and hook.source == "generated-config" and
            isinstance(path, str) and bool(path) and hook.key == ("generated-config", path))


def prepared_hooks(container: "BaseContainer") -> "tuple[Hook, ...]":
    """Return this container's exact generated-file hooks in registry order."""
    from .lifecycle.hooks import HookPhase

    return tuple(hook for hook in container.hooks.iter_phase(HookPhase.BEFORE_START)
                 if _is_prepared_hook(hook))


def prepared_paths(container: "BaseContainer") -> "tuple[Path, ...]":
    """Return only destinations declared through this container's file hooks."""
    from pathlib import Path

    return tuple(Path(hook.metadata["path"]) for hook in prepared_hooks(container))


def stage_files(container: "BaseContainer", files: "Mapping[str, str]", *,
                revision: "str | None" = None, mode: int = 0o600, group: "int | None" = None) -> "Path":
    """Materialize immutable inputs; no current pointer or service is changed."""
    from contextlib import ExitStack
    import shutil
    import tempfile
    from pathlib import Path, PurePosixPath

    root = container.get_app_path("generated")
    destination = prepared_files_path(root, files, revision=revision, mode=mode, group=group)
    with ExitStack() as stack:
        if revision is not None:
            stack.enter_context(container.manager.environ.locks.file_lock(root))
        conflict = ("Prepared file revision conflicts with existing content or permissions: "
                    if revision is not None else "Prepared file tree was modified: ") + str(destination)
        directories = {"."}
        for name in files:
            directories.update(str(parent) for parent in PurePosixPath(name).parents)
        directory_mode = 0o755
        root = destination.parent
        if destination.is_symlink():
            raise ContainerError(conflict)
        if destination.exists():
            entries = tuple(destination.rglob("*"))
            actual = {path.relative_to(destination).as_posix() for path in entries}
            expected = set(files) | (directories - {"."})
            if (actual != expected or
                    any(path.is_symlink() for path in entries) or any(
                        not (destination / name).is_dir() or
                        stat.S_IMODE((destination / name).stat().st_mode) != directory_mode
                        for name in directories)):
                raise ContainerError(conflict)
            if any(
                    not (destination / name).is_file() or
                    (destination / name).read_bytes() != content.encode("utf-8") or
                    stat.S_IMODE((destination / name).stat().st_mode) != mode or
                    (group is not None and (destination / name).stat().st_gid != group)
                    for name, content in files.items()):
                raise ContainerError(conflict)
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
        tree = os.path.relpath(str(destination), str(container.manager.data_path))
        records = {
            os.path.relpath(str(destination / name), str(container.manager.data_path)): {
                "kind": "generated-config", "container": container.name,
                "sha256": sha256_of(content), "producers": producers,
                "tree": tree,
            } for name, content in files.items()
        }
        records[tree] = {
            "kind": "generated-config-tree", "container": container.name,
            "producers": producers, "tree": tree,
        }
        container.manager.artifact_index.record(records)
        return destination


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


def _indexed_sources(manager: "ContainerManager") -> "dict[str, tuple[str, str, str]]":
    """Map exact recorded roots and files to owner, relative filename and tree."""
    from pathlib import Path

    sources = {}
    for filename, entry in manager.artifact_index.load().items():
        tree, owner = entry.get("tree"), entry.get("container")
        kind = entry.get("kind")
        if (kind not in ("generated-config", "generated-config-tree") or not isinstance(tree, str) or
                not tree or not isinstance(owner, str) or not owner):
            continue
        path, file_path = Path(tree), Path(filename)
        if (path.is_absolute() or ".." in path.parts or file_path.is_absolute() or
                ".." in file_path.parts):
            continue
        if (kind == "generated-config-tree" and file_path != path or
                kind == "generated-config" and path not in file_path.parents):
            continue
        root = str(Path(manager.data_path) / path)
        entries = {root: (owner, "", root)}
        if kind == "generated-config":
            source = str(Path(manager.data_path) / file_path)
            entries[source] = (owner, file_path.relative_to(path).as_posix(), root)
        for destination, identity in entries.items():
            if destination in sources and sources[destination] != identity:
                raise ContainerError("Conflicting generated file provenance for " + destination)
            sources[destination] = identity
    return sources


def _declared_sources(context: "OperationContext") -> "dict[str, set[tuple[str, str]]]":
    from pathlib import Path

    declared = {}
    for container in context.project_containers:
        for hook in prepared_hooks(container):
            root = Path(hook.metadata["path"])
            declared.setdefault(str(root), set()).add((container.name, ""))
            for name in hook.metadata.get("files", ()):
                path = root / name
                declared.setdefault(str(path), set()).add((container.name, path.relative_to(root).as_posix()))
    return declared


def consumer_config_changes(context: "OperationContext", model: dict,
                            models: "AppliedServiceModels", services: "Iterable[str]") -> "tuple[str, ...]":
    """Select generated-source changes before inspecting unrelated consumer build inputs."""
    declared = _declared_sources(context)
    changed = []
    for service in dict.fromkeys(services):
        if service not in model["services"]:
            continue
        current = model["services"][service]
        mounts = current.get("volumes", ())
        generated = [(position, mount) for position, mount in enumerate(mounts)
                     if isinstance(mount, dict) and mount.get("type") == "bind" and
                     mount.get("source") in declared]
        if not generated:
            continue
        previous = models.previous_model(service)
        if previous is None or service not in previous["services"]:
            raise ContainerError("No applied Compose model for configuration consumer {}; "
                                 "explicitly select its container".format(service))
        old_mounts = previous["services"][service].get("volumes", ())
        if all(position < len(old_mounts) and isinstance(old_mounts[position], dict) and
               old_mounts[position].get("source") == mount["source"] for position, mount in generated):
            continue
        changed.append(service)
    return tuple(changed)


def validate_consumer_config(context: "OperationContext", model: dict,
                             models: "AppliedServiceModels", services: "Iterable[str]") -> "tuple[str, ...]":
    """Permit only exact, declared generated-tree source updates for extra consumers."""
    from copy import deepcopy

    declared = _declared_sources(context)
    indexed = _indexed_sources(models.manager)
    changed = []
    for service in consumer_config_changes(context, model, models, services):
        current = model["services"][service]
        previous = models.previous_model(service)
        old = previous["services"][service]
        old_mounts = old.get("volumes", ())
        candidate = deepcopy(current)
        new_mounts = candidate.get("volumes", ())
        updated = False
        if len(old_mounts) == len(new_mounts):
            for before, after in zip(old_mounts, new_mounts):
                if not isinstance(before, dict) or not isinstance(after, dict):
                    continue
                old_source, new_source = before.get("source"), after.get("source")
                if (before.get("type") == after.get("type") == "bind" and
                        isinstance(old_source, str) and isinstance(new_source, str) and
                        old_source != new_source and old_source in indexed and
                        indexed[old_source][:2] in declared.get(new_source, ())):
                    after["source"] = old_source
                    updated = True
        if candidate != old or _service_support(model, service) != _service_support(previous, service):
            raise ContainerError("Configuration consumer {} has changes outside its declared generated files; "
                                 "explicitly select its container".format(service))
        if updated:
            changed.append(service)
    return tuple(changed)


def prune_prepared_files(context: "OperationContext", models: "AppliedServiceModels") -> None:
    """Retain exact applied and rollback tree sources; preserve uncertain legacy inputs."""
    import shutil
    from pathlib import Path

    running = set(context.initial_runtime_state.running_services) if context.initial_runtime_state else set()
    unapplied = running - set(context.target_services or ())
    if context.is_full_project:
        declared = {service for container in context.project_containers for service in container.services}
        unapplied.intersection_update(declared)
    if models.untracked_services.intersection(unapplied):
        return
    sources = set()
    for collection, read in ((models.current, models.current_model), (models.previous, models.previous_model)):
        for service in collection:
            spec = read(service)["services"][service]
            for mount in spec.get("volumes", ()):
                if isinstance(mount, dict) and mount.get("type") == "bind" and isinstance(mount.get("source"), str):
                    sources.add(mount["source"])
    indexed = _indexed_sources(models.manager)
    for name, candidate in context.prepared_dirs.items():
        root = candidate.parent
        known = {Path(tree) for owner, _, tree in indexed.values() if owner == name and Path(tree).parent == root}
        # A legacy mount can refer to current, a child file, or the whole root.
        # Preserve that root instead of inferring what a running service reads.
        if any(source not in indexed and
               (Path(source) == root or root in Path(source).parents or Path(source) in root.parents)
               for source in sources):
            continue
        keep = {candidate} | {Path(indexed[source][2]) for source in sources if source in indexed}
        for path in known - keep:
            identifier = path.name[len("revision-"):] if path.name.startswith("revision-") else path.name
            if (path.is_symlink() or not path.is_dir() or len(identifier) != 64 or
                    any(char not in "0123456789abcdef" for char in identifier)):
                continue
            index = models.manager.artifact_index
            tree = os.path.relpath(str(path), str(models.manager.data_path))
            removed = tuple(key for key, entry in index.load().items()
                            if entry.get("kind") in ("generated-config", "generated-config-tree") and
                            entry.get("container") == name and entry.get("tree") == tree)
            shutil.rmtree(str(path))
            index.record({}, remove=removed)


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
