#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable restore value contracts and canonical manifest codec."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from ...core import ImmutableJsonMapping, JsonValue, canonical_json_bytes, normalize_json_value

_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX_KEY = re.compile(r"[0-9a-f]{64}")
_FACT_KEY = re.compile(r"[0-9a-f]{64}/[0-9]{20}")
_SPACES = frozenset({"records", "aliases", "facts", "sequences", "operations", "control"})
_STATE_RESOURCES = frozenset(
    {
        "state:conversation",
        "state:execution",
        "state:memory",
        "state:artifact",
        "state:task",
        "state:evaluation",
        "state:recovery",
    }
)

PathPlatform = Literal["posix", "windows"]
DependencyKind = Literal["binding", "object", "effect", "restore-evidence"]


def _require_string(value: object, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError(f"{field} must be a{' non-empty' if nonempty else ''} string")
    return value


def _require_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _require_sha256(value: object, field: str) -> str:
    text = _require_string(value, field)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _require_object(value: object, field: str) -> dict[str, JsonValue]:
    normalized = normalize_json_value(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{field} must be an object")
    return normalized


def _require_list(value: object, field: str) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return cast(list[JsonValue], value)


def _require_fields(value: Mapping[str, object], *, expected: frozenset[str], field: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{field} fields mismatch: missing={missing}, unknown={unknown}")


def _immutable_object(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return ImmutableJsonMapping(value)


def _canonical_groups(groups: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    normalized: list[tuple[str, ...]] = []
    seen: set[str] = set()
    for raw_group in groups:
        group = tuple(sorted(_require_string(role, "group role") for role in raw_group))
        if not group:
            raise ValueError("restore group cannot be empty")
        if len(group) != len(set(group)):
            raise ValueError("restore group cannot contain duplicate roles")
        duplicate = seen.intersection(group)
        if duplicate:
            raise ValueError(f"restore role appears in multiple groups: {sorted(duplicate)}")
        seen.update(group)
        normalized.append(group)
    return tuple(sorted(normalized))


def _json_groups(groups: Sequence[Sequence[str]]) -> list[JsonValue]:
    return [list(group) for group in groups]


@dataclass(frozen=True, slots=True)
class Locator:
    resource: str
    space: str
    key: str

    def __post_init__(self) -> None:
        if self.resource not in _STATE_RESOURCES:
            raise ValueError("locator resource must identify one RuntimeState domain")
        if self.space not in _SPACES:
            raise ValueError("locator space is invalid")
        if self.space == "facts":
            if _FACT_KEY.fullmatch(self.key) is None:
                raise ValueError("fact locator key is invalid")
        elif self.space == "control":
            if (
                not self.key
                or self.key.startswith("/")
                or "\\" in self.key
                or any(part in {"", ".", ".."} for part in self.key.split("/"))
            ):
                raise ValueError("control locator key is invalid")
        elif _HEX_KEY.fullmatch(self.key) is None:
            raise ValueError("locator key must be a lowercase 64-hex logical key")

    def to_json(self) -> dict[str, JsonValue]:
        return {"resource": self.resource, "space": self.space, "key": self.key}

    @classmethod
    def from_json(cls, value: object) -> "Locator":
        obj = _require_object(value, "locator")
        _require_fields(obj, expected=frozenset({"resource", "space", "key"}), field="locator")
        return cls(
            _require_string(obj["resource"], "locator.resource"),
            _require_string(obj["space"], "locator.space"),
            _require_string(obj["key"], "locator.key"),
        )


@dataclass(frozen=True, slots=True)
class Cut:
    resource: str
    count: int
    digest: str
    generation: int | None

    def __post_init__(self) -> None:
        _require_string(self.resource, "cut.resource")
        _require_int(self.count, "cut.count", minimum=0)
        _require_sha256(self.digest, "cut.digest")
        if self.generation is not None:
            _require_int(self.generation, "cut.generation", minimum=0)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "resource": self.resource,
            "count": self.count,
            "digest": self.digest,
            "generation": self.generation,
        }

    @classmethod
    def from_json(cls, value: object) -> "Cut":
        obj = _require_object(value, "cut")
        _require_fields(
            obj,
            expected=frozenset({"resource", "count", "digest", "generation"}),
            field="cut",
        )
        generation = obj["generation"]
        return cls(
            _require_string(obj["resource"], "cut.resource"),
            _require_int(obj["count"], "cut.count", minimum=0),
            _require_sha256(obj["digest"], "cut.digest"),
            None if generation is None else _require_int(generation, "cut.generation", minimum=0),
        )


@dataclass(frozen=True, slots=True)
class WorkspaceLocation:
    platform: PathPlatform
    root: str

    def __post_init__(self) -> None:
        if self.platform not in {"posix", "windows"}:
            raise ValueError("workspace location platform is invalid")
        _require_string(self.root, "workspace_location.root")

    def to_json(self) -> dict[str, JsonValue]:
        return {"platform": self.platform, "root": self.root}

    @classmethod
    def from_json(cls, value: object) -> "WorkspaceLocation":
        obj = _require_object(value, "workspace_location")
        _require_fields(obj, expected=frozenset({"platform", "root"}), field="workspace_location")
        platform = _require_string(obj["platform"], "workspace_location.platform")
        if platform not in {"posix", "windows"}:
            raise ValueError("workspace location platform is invalid")
        return cls(cast(PathPlatform, platform), _require_string(obj["root"], "workspace_location.root"))


@dataclass(frozen=True, slots=True)
class PathOrigin:
    version: int
    namespace: str
    platform: PathPlatform
    root: str

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("PathOrigin version must be 1")
        _require_string(self.namespace, "path_origin.namespace")
        if self.platform not in {"posix", "windows"}:
            raise ValueError("path_origin.platform is invalid")
        _require_string(self.root, "path_origin.root")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "version": 1,
            "namespace": self.namespace,
            "platform": self.platform,
            "root": self.root,
        }

    @classmethod
    def from_json(cls, value: object) -> "PathOrigin":
        obj = _require_object(value, "path_origin")
        _require_fields(
            obj,
            expected=frozenset({"version", "namespace", "platform", "root"}),
            field="path_origin",
        )
        platform = _require_string(obj["platform"], "path_origin.platform")
        if platform not in {"posix", "windows"}:
            raise ValueError("path_origin.platform is invalid")
        return cls(
            _require_int(obj["version"], "path_origin.version"),
            _require_string(obj["namespace"], "path_origin.namespace"),
            cast(PathPlatform, platform),
            _require_string(obj["root"], "path_origin.root"),
        )


@dataclass(frozen=True, slots=True)
class LogicalPath:
    namespace: str
    relative: str

    def __post_init__(self) -> None:
        _require_string(self.namespace, "logical_path.namespace")
        if (
            not isinstance(self.relative, str)
            or not self.relative
            or self.relative.startswith("/")
            or "\\" in self.relative
            or any(part in {"", ".", ".."} for part in self.relative.split("/"))
        ):
            raise ValueError("logical_path.relative must be a normalized relative POSIX path")

    def to_json(self) -> dict[str, JsonValue]:
        return {"namespace": self.namespace, "relative": self.relative}


@dataclass(frozen=True, slots=True)
class PathEvidence:
    version: int
    at: Locator
    value_pointer: str
    source_value_digest: str
    origin: PathOrigin
    relative: str

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("PathEvidence version must be 1")
        if not isinstance(self.at, Locator) or not isinstance(self.origin, PathOrigin):
            raise TypeError("PathEvidence requires Locator and PathOrigin values")
        if not isinstance(self.value_pointer, str) or (
            self.value_pointer and not self.value_pointer.startswith("/")
        ):
            raise ValueError("path_evidence.value_pointer must be a JSON Pointer")
        _require_sha256(self.source_value_digest, "path_evidence.source_value_digest")
        LogicalPath(self.origin.namespace, self.relative)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "version": 1,
            "at": self.at.to_json(),
            "value_pointer": self.value_pointer,
            "source_value_digest": self.source_value_digest,
            "origin": self.origin.to_json(),
            "relative": self.relative,
        }

    @classmethod
    def from_json(cls, value: object) -> "PathEvidence":
        obj = _require_object(value, "path_evidence")
        _require_fields(
            obj,
            expected=frozenset(
                {"version", "at", "value_pointer", "source_value_digest", "origin", "relative"}
            ),
            field="path_evidence",
        )
        return cls(
            _require_int(obj["version"], "path_evidence.version"),
            Locator.from_json(obj["at"]),
            _require_string(obj["value_pointer"], "path_evidence.value_pointer", nonempty=False),
            _require_sha256(obj["source_value_digest"], "path_evidence.source_value_digest"),
            PathOrigin.from_json(obj["origin"]),
            _require_string(obj["relative"], "path_evidence.relative"),
        )


@dataclass(frozen=True, slots=True)
class Dependency:
    kind: DependencyKind
    locator: Locator
    digest: str

    def __post_init__(self) -> None:
        if self.kind not in {"binding", "object", "effect", "restore-evidence"}:
            raise ValueError("restore dependency kind is invalid")
        if not isinstance(self.locator, Locator):
            raise TypeError("restore dependency locator must be Locator")
        _require_sha256(self.digest, "dependency.digest")

    def to_json(self) -> dict[str, JsonValue]:
        return {"kind": self.kind, "locator": self.locator.to_json(), "digest": self.digest}

    @classmethod
    def from_json(cls, value: object) -> "Dependency":
        obj = _require_object(value, "dependency")
        _require_fields(obj, expected=frozenset({"kind", "locator", "digest"}), field="dependency")
        kind = _require_string(obj["kind"], "dependency.kind")
        if kind not in {"binding", "object", "effect", "restore-evidence"}:
            raise ValueError("dependency.kind is invalid")
        return cls(
            cast(DependencyKind, kind),
            Locator.from_json(obj["locator"]),
            _require_sha256(obj["digest"], "dependency.digest"),
        )


@dataclass(frozen=True, slots=True)
class LegacyStorageEvidence:
    checkpoint: Locator
    checkpoint_digest: str
    revision: int
    old_handoff_digest: str
    source_v6: Mapping[str, JsonValue]
    source_groups: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, Locator):
            raise TypeError("legacy checkpoint must be Locator")
        _require_sha256(self.checkpoint_digest, "legacy.checkpoint_digest")
        _require_int(self.revision, "legacy.revision", minimum=0)
        _require_sha256(self.old_handoff_digest, "legacy.old_handoff_digest")
        object.__setattr__(
            self,
            "source_v6",
            _immutable_object(_require_object(self.source_v6, "legacy.source_v6")),
        )
        object.__setattr__(self, "source_groups", _canonical_groups(self.source_groups))

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "checkpoint": self.checkpoint.to_json(),
            "checkpoint_digest": self.checkpoint_digest,
            "revision": self.revision,
            "old_handoff_digest": self.old_handoff_digest,
            "source_v6": dict(self.source_v6),
            "source_groups": _json_groups(self.source_groups),
        }

    @classmethod
    def from_json(cls, value: object) -> "LegacyStorageEvidence":
        obj = _require_object(value, "legacy_storage_evidence")
        _require_fields(
            obj,
            expected=frozenset(
                {
                    "checkpoint",
                    "checkpoint_digest",
                    "revision",
                    "old_handoff_digest",
                    "source_v6",
                    "source_groups",
                }
            ),
            field="legacy_storage_evidence",
        )
        return cls(
            Locator.from_json(obj["checkpoint"]),
            _require_sha256(obj["checkpoint_digest"], "legacy.checkpoint_digest"),
            _require_int(obj["revision"], "legacy.revision", minimum=0),
            _require_sha256(obj["old_handoff_digest"], "legacy.old_handoff_digest"),
            _require_object(obj["source_v6"], "legacy.source_v6"),
            _parse_groups(obj["source_groups"], "legacy.source_groups"),
        )


@dataclass(frozen=True, slots=True)
class RestoreResource:
    role: str
    backend: str
    retention: Literal["durable"]
    layout_version: int
    source_locator: Mapping[str, JsonValue]
    relative_layout: tuple[str, ...]
    cut: Cut

    def __post_init__(self) -> None:
        _require_string(self.role, "restore_resource.role")
        _require_string(self.backend, "restore_resource.backend")
        if self.retention != "durable":
            raise ValueError("restore_resource.retention must be durable")
        _require_int(self.layout_version, "restore_resource.layout_version", minimum=1)
        object.__setattr__(
            self,
            "source_locator",
            _immutable_object(_require_object(self.source_locator, "restore_resource.source_locator")),
        )
        layout = tuple(
            _require_string(value, "restore_resource.relative_layout")
            for value in self.relative_layout
        )
        if len(layout) != len(set(layout)):
            raise ValueError("restore_resource.relative_layout contains duplicates")
        object.__setattr__(self, "relative_layout", layout)
        if not isinstance(self.cut, Cut):
            raise TypeError("restore_resource.cut must be Cut")

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "role": self.role,
            "backend": self.backend,
            "retention": self.retention,
            "layout_version": self.layout_version,
            "source_locator": dict(self.source_locator),
            "relative_layout": list(self.relative_layout),
            "cut": self.cut.to_json(),
        }

    @classmethod
    def from_json(cls, value: object) -> "RestoreResource":
        obj = _require_object(value, "restore_resource")
        _require_fields(
            obj,
            expected=frozenset(
                {"role", "backend", "retention", "layout_version", "source_locator", "relative_layout", "cut"}
            ),
            field="restore_resource",
        )
        retention = _require_string(obj["retention"], "restore_resource.retention")
        if retention != "durable":
            raise ValueError("restore_resource.retention must be durable")
        return cls(
            _require_string(obj["role"], "restore_resource.role"),
            _require_string(obj["backend"], "restore_resource.backend"),
            "durable",
            _require_int(obj["layout_version"], "restore_resource.layout_version", minimum=1),
            _require_object(obj["source_locator"], "restore_resource.source_locator"),
            tuple(
                _require_string(item, "restore_resource.relative_layout")
                for item in _require_list(obj["relative_layout"], "restore_resource.relative_layout")
            ),
            Cut.from_json(obj["cut"]),
        )


@dataclass(frozen=True, slots=True)
class RestoreGroups:
    state: tuple[tuple[str, ...], ...]
    objects: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        state = _canonical_groups(self.state)
        objects = _canonical_groups(self.objects)
        overlap = {role for group in state for role in group}.intersection(
            role for group in objects for role in group
        )
        if overlap:
            raise ValueError(f"restore roles cannot be both state and object roles: {sorted(overlap)}")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "objects", objects)

    def to_json(self) -> dict[str, JsonValue]:
        return {"state": _json_groups(self.state), "objects": _json_groups(self.objects)}

    @classmethod
    def from_json(cls, value: object) -> "RestoreGroups":
        obj = _require_object(value, "restore_groups")
        _require_fields(obj, expected=frozenset({"state", "objects"}), field="restore_groups")
        return cls(
            _parse_groups(obj["state"], "restore_groups.state"),
            _parse_groups(obj["objects"], "restore_groups.objects"),
        )

    def roles(self) -> frozenset[str]:
        return frozenset(role for group in (*self.state, *self.objects) for role in group)


@dataclass(frozen=True, slots=True)
class RestoreManifest:
    version: int
    namespace: str
    tenant_id: str
    source_workspace: WorkspaceLocation
    resources: tuple[RestoreResource, ...]
    groups: RestoreGroups
    legacy_contracts: tuple[LegacyStorageEvidence, ...] = ()
    dependencies: tuple[Dependency, ...] = ()
    path_evidence: tuple[PathEvidence, ...] = ()

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("RestoreManifest version must be 1")
        _require_string(self.namespace, "restore_manifest.namespace")
        _require_string(self.tenant_id, "restore_manifest.tenant_id")
        if not isinstance(self.source_workspace, WorkspaceLocation):
            raise TypeError("restore_manifest.source_workspace must be WorkspaceLocation")
        resources = tuple(sorted(self.resources, key=lambda item: item.role))
        if any(not isinstance(item, RestoreResource) for item in resources):
            raise TypeError("restore_manifest.resources must contain RestoreResource")
        roles = tuple(item.role for item in resources)
        if len(roles) != len(set(roles)):
            raise ValueError("restore_manifest.resources contains duplicate roles")
        if not isinstance(self.groups, RestoreGroups):
            raise TypeError("restore_manifest.groups must be RestoreGroups")
        if self.groups.roles() != frozenset(roles):
            raise ValueError("restore groups must cover exactly the manifest resource roles")
        legacy = tuple(
            sorted(self.legacy_contracts, key=lambda item: _locator_sort_key(item.checkpoint))
        )
        dependencies = tuple(
            sorted(
                self.dependencies,
                key=lambda item: (item.kind, *_locator_sort_key(item.locator), item.digest),
            )
        )
        paths = tuple(
            sorted(
                self.path_evidence,
                key=lambda item: (
                    *_locator_sort_key(item.at),
                    item.value_pointer,
                    item.source_value_digest,
                ),
            )
        )
        if any(not isinstance(item, LegacyStorageEvidence) for item in legacy):
            raise TypeError("restore_manifest.legacy_contracts contains invalid values")
        if any(not isinstance(item, Dependency) for item in dependencies):
            raise TypeError("restore_manifest.dependencies contains invalid values")
        if any(not isinstance(item, PathEvidence) for item in paths):
            raise TypeError("restore_manifest.path_evidence contains invalid values")
        if len(legacy) != len(
            set((item.checkpoint, item.checkpoint_digest, item.revision) for item in legacy)
        ):
            raise ValueError("restore_manifest.legacy_contracts contains duplicates")
        if len(dependencies) != len(
            set((item.kind, item.locator, item.digest) for item in dependencies)
        ):
            raise ValueError("restore_manifest.dependencies contains duplicates")
        if len(paths) != len(
            set((item.at, item.value_pointer, item.source_value_digest) for item in paths)
        ):
            raise ValueError("restore_manifest.path_evidence contains duplicates")
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "legacy_contracts", legacy)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "path_evidence", paths)

    def to_json(self) -> dict[str, JsonValue]:
        return {
            "version": 1,
            "namespace": self.namespace,
            "tenant_id": self.tenant_id,
            "source_workspace": self.source_workspace.to_json(),
            "resources": [item.to_json() for item in self.resources],
            "groups": self.groups.to_json(),
            "legacy_contracts": [item.to_json() for item in self.legacy_contracts],
            "dependencies": [item.to_json() for item in self.dependencies],
            "path_evidence": [item.to_json() for item in self.path_evidence],
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_json())

    def digest(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> "RestoreManifest":
        if not isinstance(data, bytes):
            raise TypeError("RestoreManifest.from_bytes requires bytes")
        try:
            value = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicate_keys)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid restore manifest JSON") from error
        obj = _require_object(value, "restore_manifest")
        _require_fields(
            obj,
            expected=frozenset(
                {
                    "version",
                    "namespace",
                    "tenant_id",
                    "source_workspace",
                    "resources",
                    "groups",
                    "legacy_contracts",
                    "dependencies",
                    "path_evidence",
                }
            ),
            field="restore_manifest",
        )
        manifest = cls(
            _require_int(obj["version"], "restore_manifest.version"),
            _require_string(obj["namespace"], "restore_manifest.namespace"),
            _require_string(obj["tenant_id"], "restore_manifest.tenant_id"),
            WorkspaceLocation.from_json(obj["source_workspace"]),
            tuple(
                RestoreResource.from_json(item)
                for item in _require_list(obj["resources"], "restore_manifest.resources")
            ),
            RestoreGroups.from_json(obj["groups"]),
            tuple(
                LegacyStorageEvidence.from_json(item)
                for item in _require_list(
                    obj["legacy_contracts"], "restore_manifest.legacy_contracts"
                )
            ),
            tuple(
                Dependency.from_json(item)
                for item in _require_list(obj["dependencies"], "restore_manifest.dependencies")
            ),
            tuple(
                PathEvidence.from_json(item)
                for item in _require_list(obj["path_evidence"], "restore_manifest.path_evidence")
            ),
        )
        if manifest.to_bytes() != data:
            raise ValueError("restore manifest is not canonical JSON")
        return manifest


@dataclass(frozen=True, slots=True)
class RestorePlan:
    manifest: RestoreManifest
    expected_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RestoreManifest):
            raise TypeError("RestorePlan.manifest must be RestoreManifest")
        _require_sha256(self.expected_digest, "restore_plan.expected_digest")

    def verify_digest(self) -> None:
        if self.manifest.digest() != self.expected_digest:
            raise ValueError("restore manifest digest does not match expected_digest")


def _parse_groups(value: object, field: str) -> tuple[tuple[str, ...], ...]:
    groups: list[tuple[str, ...]] = []
    for raw_group in _require_list(value, field):
        groups.append(
            tuple(
                _require_string(item, f"{field} role")
                for item in _require_list(raw_group, f"{field} group")
            )
        )
    return _canonical_groups(groups)


def _locator_sort_key(locator: Locator) -> tuple[str, str, str]:
    return (locator.resource, locator.space, locator.key)


def _reject_duplicate_keys(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    value: dict[str, JsonValue] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


__all__ = [
    "Cut",
    "Dependency",
    "DependencyKind",
    "LegacyStorageEvidence",
    "Locator",
    "LogicalPath",
    "PathEvidence",
    "PathOrigin",
    "PathPlatform",
    "RestoreGroups",
    "RestoreManifest",
    "RestorePlan",
    "RestoreResource",
    "WorkspaceLocation",
]
