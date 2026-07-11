#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repository Manifest (Spec Part III): a static ``.linktools.json`` a
repository author may place at its root to declare who they are, which
cntr/Python/Docker/Compose versions they require, and free-form metadata.

Security boundary (section 22): standard JSON only, max 1 MiB, root must be
an object, no JSON5/YAML/comments/trailing-commas/Jinja/env-interpolation/
expression-execution, ``$schema`` is never downloaded. This is validated
*before* a repository's ``container.py`` is imported -- but the manifest is
not a sandbox or a trust boundary: once loadable, the repository's Python is
still fully-trusted code.
"""
import hashlib
import json
import os
import platform
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from ..container import ContainerError

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import PathType
    from ..manager import ContainerManager

WARN = "WARN"
INFO = "INFO"

_MAX_MANIFEST_BYTES = 1024 * 1024  # 1 MiB
_HOST_REQUIREMENT_KEYS = ("linktools-cntr", "python")
_RUNTIME_REQUIREMENT_KEYS = ("docker-engine", "docker-compose")


class RepositoryManifestError(ContainerError):
    pass


class RepositoryManifestInvalid(RepositoryManifestError):
    pass


class RepositorySchemaUnsupported(RepositoryManifestError):
    pass


class RepositoryIncompatible(RepositoryManifestError):
    pass


@dataclass(frozen=True)
class RepositoryManifest:
    schema_version: int
    kind: str
    name: "str | None" = None
    version: "str | None" = None
    description: "str | None" = None
    requires: "dict[str, str]" = field(default_factory=dict)
    metadata: "dict[str, Any]" = field(default_factory=dict)
    extensions: "dict[str, Any]" = field(default_factory=dict)


@dataclass(frozen=True)
class CompatibilityIssue:
    key: str
    required: str
    actual: "str | None"
    severity: str
    message: str


@dataclass(frozen=True)
class ContainerRepositoryContext:
    """Private container-side record of where a container came from.

    Not exposed as public BaseContainer API (Spec section 27) -- Lock/Plan
    (later phases) read it directly off ``container._repository``.
    """
    url: "str | None"
    root_path: "PathType | None"
    manifest: "RepositoryManifest | None"
    builtin: bool


def _fail(exc_type, message: str):
    raise exc_type(message)


class RepositoryManifestService:
    """Load, validate and compare a repository's ``.linktools.json``."""

    file_name = ".linktools.json"
    supported_schema_versions = (1,)

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager

    def load(self, repo_path: "PathType") -> "RepositoryManifest | None":
        """Load and statically validate the manifest at ``repo_path``.

        Returns ``None`` (a legacy repository) if no manifest file is
        present -- that is not an error and must not warn.
        """
        path = os.path.join(str(repo_path), self.file_name)
        if not os.path.exists(path):
            return None

        size = os.path.getsize(path)
        if size > _MAX_MANIFEST_BYTES:
            _fail(RepositoryManifestInvalid,
                  f"{self.file_name} is {size} bytes, exceeding the {_MAX_MANIFEST_BYTES}-byte limit")

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        if not text.strip():
            _fail(RepositoryManifestInvalid, f"{self.file_name} is empty")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RepositoryManifestInvalid(f"{self.file_name} is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise RepositoryManifestInvalid(f"{self.file_name} root must be a JSON object")

        manifest = self._parse(data)
        self.validate_static(manifest)
        return manifest

    def _parse(self, data: "dict[str, Any]") -> "RepositoryManifest":
        if "schema_version" not in data:
            raise RepositoryManifestInvalid(f"{self.file_name} is missing required field 'schema_version'")
        schema_version = data["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise RepositoryManifestInvalid("'schema_version' must be an integer")

        if "kind" not in data:
            raise RepositoryManifestInvalid(f"{self.file_name} is missing required field 'kind'")
        kind = data["kind"]

        name = data.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise RepositoryManifestInvalid("'name' must be a non-empty string when present")

        version = data.get("version")
        if version is not None and not isinstance(version, str):
            raise RepositoryManifestInvalid("'version' must be a string when present")

        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise RepositoryManifestInvalid("'description' must be a string when present")

        requires = data.get("requires", {})
        if not isinstance(requires, dict) or not all(isinstance(v, str) for v in requires.values()):
            raise RepositoryManifestInvalid("'requires' must be an object of string specifiers")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise RepositoryManifestInvalid("'metadata' must be an object when present")

        extensions = data.get("extensions", {})
        if not isinstance(extensions, dict):
            raise RepositoryManifestInvalid("'extensions' must be an object when present")

        known_keys = {"$schema", "schema_version", "kind", "name", "version",
                      "description", "requires", "metadata", "extensions"}
        unknown = set(data.keys()) - known_keys
        if unknown:
            raise RepositoryManifestInvalid(f"Unknown top-level field(s): {', '.join(sorted(unknown))}")

        return RepositoryManifest(
            schema_version=schema_version,
            kind=kind,
            name=name,
            version=version,
            description=description,
            requires=dict(requires),
            metadata=dict(metadata),
            extensions=dict(extensions),
        )

    def validate_static(self, manifest: "RepositoryManifest") -> None:
        """Schema/kind/version validation that needs no host/runtime info."""
        if manifest.schema_version not in self.supported_schema_versions:
            raise RepositorySchemaUnsupported(
                f"Unsupported manifest schema_version {manifest.schema_version}; "
                f"this cntr supports {self.supported_schema_versions}"
            )
        if manifest.kind != "linktools-cntr-repository":
            raise RepositoryManifestInvalid(
                f"Unsupported manifest kind {manifest.kind!r}; expected 'linktools-cntr-repository'"
            )
        for key, specifier in manifest.requires.items():
            try:
                SpecifierSet(specifier)
            except InvalidSpecifier as exc:
                raise RepositoryManifestInvalid(
                    f"requires.{key} is not a valid PEP 440 specifier: {specifier!r}"
                ) from exc
        if manifest.version is not None:
            try:
                Version(manifest.version)
            except InvalidVersion as exc:
                raise RepositoryManifestInvalid(f"'version' is not PEP 440 compatible: {manifest.version!r}") from exc

    def _check(self, manifest: "RepositoryManifest", keys, actual_of) -> "list[CompatibilityIssue]":
        issues: "list[CompatibilityIssue]" = []
        for key in keys:
            if key not in manifest.requires:
                continue
            specifier = manifest.requires[key]
            actual = actual_of(key)
            try:
                specifier_set = SpecifierSet(specifier)
            except InvalidSpecifier:
                issues.append(CompatibilityIssue(
                    key=key, required=specifier, actual=actual, severity=WARN,
                    message=f"requires.{key} is not a valid PEP 440 specifier: {specifier!r}",
                ))
                continue
            if actual is None:
                issues.append(CompatibilityIssue(
                    key=key, required=specifier, actual=None, severity=WARN,
                    message=f"{key} version could not be determined; required {specifier}",
                ))
                continue
            try:
                satisfied = specifier_set.contains(Version(actual), prereleases=True)
            except InvalidVersion:
                issues.append(CompatibilityIssue(
                    key=key, required=specifier, actual=actual, severity=WARN,
                    message=f"{key} version {actual!r} is not PEP 440 comparable",
                ))
                continue
            if not satisfied:
                issues.append(CompatibilityIssue(
                    key=key, required=specifier, actual=actual, severity=WARN,
                    message=f"{key} {actual} does not satisfy required {specifier}",
                ))
        return issues

    def check_host_requirements(self, manifest: "RepositoryManifest") -> "list[CompatibilityIssue]":
        """linktools-cntr / python -- checked before a repo's Python is imported."""
        from ...capabilities.cntr import __cap_cntr__

        def actual_of(key: str) -> "str | None":
            if key == "linktools-cntr":
                return __cap_cntr__.version
            if key == "python":
                return platform.python_version()
            return None

        return self._check(manifest, _HOST_REQUIREMENT_KEYS, actual_of)

    def check_runtime_requirements(self, manifest: "RepositoryManifest") -> "list[CompatibilityIssue]":
        """docker-engine / docker-compose -- checked before an actual Compose
        operation (doctor, repo status --runtime, plan, up/restart/down/config
        preflight), never required just to add or load a repository."""

        def actual_of(key: str) -> "str | None":
            inspector = getattr(self.manager, "docker_inspector", None)
            if inspector is None:
                return None
            try:
                if key == "docker-engine":
                    engine = inspector.get_engine_version()
                    return engine.server or engine.client if engine else None
                if key == "docker-compose":
                    return inspector.get_compose_version()
            except Exception:  # noqa: BLE001 - a version probe must never crash the caller
                return None
            return None

        return self._check(manifest, _RUNTIME_REQUIREMENT_KEYS, actual_of)

    def ensure_loadable(self, manifest: "RepositoryManifest | None") -> None:
        """Raise if ``manifest`` (already statically valid) fails a host
        requirement -- called right before importing container.py."""
        if manifest is None:
            return  # legacy repository
        issues = self.check_host_requirements(manifest)
        if issues:
            details = "; ".join(issue.message for issue in issues)
            raise RepositoryIncompatible(f"Repository is incompatible with this host: {details}")

    def unknown_requirement_keys(self, manifest: "RepositoryManifest") -> "list[str]":
        """``requires`` keys outside the v1 standard set. Kept, not an
        error -- Doctor surfaces them as INFO (Spec section 21)."""
        known = set(_HOST_REQUIREMENT_KEYS) | set(_RUNTIME_REQUIREMENT_KEYS)
        return sorted(key for key in manifest.requires if key not in known)


def describe_repository(
        manager: "ContainerManager", url: str, meta: "dict[str, Any]", check_runtime: bool = False,
) -> "dict[str, Any]":
    """Read-only repository status/validation summary for ``repo status``/
    ``repo validate`` (Spec section 28): manifest presence, name/version,
    required cntr/Python (and, opt-in, docker-engine/docker-compose),
    compatibility result, manifest hash, Git revision and dirty state.
    Never imports the repository's ``container.py``.
    """
    repo_path = meta.get("repo_path")
    info: "dict[str, Any]" = dict(url=url, type=meta.get("type", "unknown"), repo_path=repo_path)

    manifest_path = os.path.join(repo_path, RepositoryManifestService.file_name) if repo_path else None
    if not manifest_path or not os.path.exists(manifest_path):
        info["manifest"] = "legacy"
        return info

    info["manifest"] = "present"
    with open(manifest_path, "rb") as f:
        info["manifest_sha256"] = hashlib.sha256(f.read()).hexdigest()

    try:
        manifest = manager.repo_manifest.load(repo_path)
    except RepositoryManifestError as exc:
        info["manifest_error"] = str(exc)
        info["compatible"] = False
        return info

    info["manifest_name"] = manifest.name
    info["manifest_version"] = manifest.version
    info["required_linktools_cntr"] = manifest.requires.get("linktools-cntr")
    info["required_python"] = manifest.requires.get("python")

    issues = list(manager.repo_manifest.check_host_requirements(manifest))
    if check_runtime:
        info["required_docker_engine"] = manifest.requires.get("docker-engine")
        info["required_docker_compose"] = manifest.requires.get("docker-compose")
        issues += manager.repo_manifest.check_runtime_requirements(manifest)

    info["compatible"] = not issues
    if issues:
        info["compatibility_issues"] = [issue.message for issue in issues]

    if info["type"] == "git" and repo_path and os.path.exists(repo_path):
        try:
            from linktools.git import GitRepository
            from dulwich.errors import NotGitRepository
            repo = GitRepository(manager.environ, repo_path)
            info["revision"] = repo.head_sha()
            info["dirty"] = repo.is_dirty()
        except NotGitRepository:
            pass
        except Exception:  # noqa: BLE001 - status must stay read-only & non-fatal
            pass

    return info
