#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic ``.linktools.json`` project manifest: file format, parsing,
static validation, and a version-requirement resolver registry.

A manifest describes a *Linktools Project* -- a directory that may contain
one or more capability "components" (``cntr``, ``ai``, ...). This module
only knows the generic envelope (top-level fields plus opaque per-component
``config``/``metadata``/``extensions`` blocks) and a generic requirement-key
grammar (``python``, ``package:<name>``, ``runtime:<name>``); it must never
import a capability package (``linktools.cntr``, ``linktools.ai``, ...) --
each capability interprets its own ``components.<name>`` block and registers
its own ``runtime:*`` resolvers.

Security boundary: standard JSON only, max 1 MiB, root must be an object, no
JSON5/YAML/comments/trailing-commas/Jinja/env-interpolation/expression
execution, ``$schema`` is never downloaded. This is validated *before* a
project's capability-specific Python is ever imported -- but the manifest is
not a sandbox or a trust boundary: once loadable, that Python is still
fully-trusted code.
"""
import json
import os
import platform
import re
import stat
from enum import Enum
from typing import TYPE_CHECKING, NamedTuple

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from ..errors import ManifestLoadError, ManifestSchemaUnsupported, ManifestValidationError

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # Python < 3.8
    from importlib_metadata import PackageNotFoundError, version as _pkg_version

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional
    from linktools.types import PathType

_MAX_MANIFEST_BYTES = 1024 * 1024  # 1 MiB
_MANIFEST_KIND = "linktools-project"

_TOP_LEVEL_FIELDS = frozenset({
    "$schema", "schema_version", "kind", "name", "version",
    "description", "requires", "components", "metadata", "extensions",
})
_COMPONENT_FIELDS = frozenset({
    "schema_version", "requires", "config", "metadata", "extensions",
})

# A requirement key is either the bare "python", or "<namespace>:<name>"
# (e.g. "package:linktools-cntr", "runtime:docker-compose") -- the namespace
# lets each capability register its own resolver prefix without colliding
# with another's.
_COMPONENT_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")
_REQUIREMENT_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9-]*:[A-Za-z0-9][A-Za-z0-9._-]*$")


def _is_valid_requirement_key(key: str) -> bool:
    return key == "python" or bool(_REQUIREMENT_KEY_PATTERN.match(key))


class ManifestComponent(NamedTuple):
    """One capability's (``cntr``, ``ai``, ...) block within a manifest.

    ``config``/``metadata``/``extensions`` are opaque to Core: only checked
    to be JSON objects, never interpreted here.
    """
    schema_version: int
    requires: "Dict[str, str]"
    config: "Dict[str, Any]"
    metadata: "Dict[str, Any]"
    extensions: "Dict[str, Any]"


class LinktoolsManifest(NamedTuple):
    schema_version: int
    kind: str
    name: "Optional[str]"
    version: "Optional[str]"
    description: "Optional[str]"
    requires: "Dict[str, str]"
    components: "Dict[str, ManifestComponent]"
    metadata: "Dict[str, Any]"
    extensions: "Dict[str, Any]"

    def get_component(self, name: str) -> "Optional[ManifestComponent]":
        return self.components.get(name)


class ManifestLoader:
    """Loads and statically validates a ``.linktools.json`` file.

    ``load()``/``loads()`` never check whether any ``requires`` entry is
    actually satisfied on this host -- only that the file is well-formed
    (see ``RequirementResolverRegistry`` for that).
    """

    file_name = ".linktools.json"
    max_bytes = _MAX_MANIFEST_BYTES
    supported_schema_versions = (1,)

    def load(self, root_path: "PathType") -> "Optional[LinktoolsManifest]":
        """``None`` means no manifest file is present -- a project without
        one is not an error, just a project that opts out of the format.
        Any other filesystem failure between here and reading the file
        (permission denied, path is a directory, disappears mid-read, a
        broken symlink, ...) is a ``ManifestLoadError``, never a raw
        ``OSError`` -- callers only ever need to catch ``ManifestError``."""
        path = os.path.join(str(root_path), self.file_name)
        try:
            stat_result = os.stat(path)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ManifestLoadError("Unable to inspect %s: %s" % (self.file_name, exc.strerror or exc)) from exc

        if not stat.S_ISREG(stat_result.st_mode):
            raise ManifestLoadError("%s must be a regular file" % self.file_name)

        if stat_result.st_size > self.max_bytes:
            raise ManifestLoadError(
                "%s is %d bytes, exceeding the %d-byte limit" % (self.file_name, stat_result.st_size, self.max_bytes)
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            raise ManifestLoadError("Unable to read %s: %s" % (self.file_name, exc.strerror or exc)) from exc
        return self.loads(text)

    def loads(self, text: str) -> "LinktoolsManifest":
        if not text.strip():
            raise ManifestLoadError("%s is empty" % self.file_name)
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ManifestLoadError("%s is not valid JSON: %s" % (self.file_name, exc))
        if not isinstance(data, dict):
            raise ManifestValidationError("%s root must be a JSON object" % self.file_name)

        manifest = self.parse(data)
        self.validate(manifest)
        return manifest

    def parse(self, data: "Dict[str, Any]") -> "LinktoolsManifest":
        unknown = set(data.keys()) - _TOP_LEVEL_FIELDS
        if unknown:
            raise ManifestValidationError("Unknown top-level field(s): %s" % ", ".join(sorted(unknown)))

        if "schema_version" not in data:
            raise ManifestValidationError("%s is missing required field 'schema_version'" % self.file_name)
        schema_version = data["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ManifestValidationError("'schema_version' must be an integer")

        if "kind" not in data:
            raise ManifestValidationError("%s is missing required field 'kind'" % self.file_name)
        kind = data["kind"]

        name = data.get("name")
        if name is not None and (not isinstance(name, str) or not name.strip()):
            raise ManifestValidationError("'name' must be a non-empty string when present")

        version = data.get("version")
        if version is not None and not isinstance(version, str):
            raise ManifestValidationError("'version' must be a string when present")

        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise ManifestValidationError("'description' must be a string when present")

        requires = self._parse_requires(data.get("requires", {}), "requires")

        components_data = data.get("components", {})
        if not isinstance(components_data, dict):
            raise ManifestValidationError("'components' must be an object when present")
        components = {
            component_id: self._parse_component(component_id, component_data)
            for component_id, component_data in components_data.items()
        }

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ManifestValidationError("'metadata' must be an object when present")

        extensions = data.get("extensions", {})
        if not isinstance(extensions, dict):
            raise ManifestValidationError("'extensions' must be an object when present")

        return LinktoolsManifest(
            schema_version=schema_version,
            kind=kind,
            name=name,
            version=version,
            description=description,
            requires=dict(requires),
            components=components,
            metadata=dict(metadata),
            extensions=dict(extensions),
        )

    def _parse_requires(self, requires: "Any", where: str) -> "Dict[str, str]":
        if not isinstance(requires, dict) or not all(isinstance(v, str) for v in requires.values()):
            raise ManifestValidationError("'%s' must be an object of string specifiers" % where)
        invalid_keys = [key for key in requires if not _is_valid_requirement_key(key)]
        if invalid_keys:
            raise ManifestValidationError(
                "'%s' has invalid key(s) (must be `python` or `<namespace>:<name>`): %s"
                % (where, ", ".join(sorted(invalid_keys)))
            )
        return dict(requires)

    def _parse_component(self, component_id: str, data: "Any") -> "ManifestComponent":
        if not _COMPONENT_ID_PATTERN.match(component_id):
            raise ManifestValidationError(
                "component id %r is invalid (must match `%s`)" % (component_id, _COMPONENT_ID_PATTERN.pattern)
            )
        if not isinstance(data, dict):
            raise ManifestValidationError("components.%s must be an object" % component_id)

        unknown = set(data.keys()) - _COMPONENT_FIELDS
        if unknown:
            raise ManifestValidationError(
                "components.%s has unknown field(s): %s" % (component_id, ", ".join(sorted(unknown)))
            )

        if "schema_version" not in data:
            raise ManifestValidationError("components.%s is missing required field 'schema_version'" % component_id)
        schema_version = data["schema_version"]
        if not isinstance(schema_version, int) or isinstance(schema_version, bool) or schema_version < 1:
            raise ManifestValidationError("components.%s.schema_version must be an integer >= 1" % component_id)

        requires = self._parse_requires(data.get("requires", {}), "components.%s.requires" % component_id)

        config = data.get("config", {})
        if not isinstance(config, dict):
            raise ManifestValidationError("components.%s.config must be an object when present" % component_id)

        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ManifestValidationError("components.%s.metadata must be an object when present" % component_id)

        extensions = data.get("extensions", {})
        if not isinstance(extensions, dict):
            raise ManifestValidationError("components.%s.extensions must be an object when present" % component_id)

        return ManifestComponent(
            schema_version=schema_version,
            requires=dict(requires),
            config=dict(config),
            metadata=dict(metadata),
            extensions=dict(extensions),
        )

    def validate(self, manifest: "LinktoolsManifest") -> None:
        """Semantic checks that don't need host/runtime info: schema
        version support, ``kind``, and PEP 440 syntax of every version/
        requirement-specifier string (top-level and per-component)."""
        if manifest.schema_version not in self.supported_schema_versions:
            raise ManifestSchemaUnsupported(
                "Unsupported manifest schema_version %r; this linktools supports %r"
                % (manifest.schema_version, self.supported_schema_versions)
            )
        if manifest.kind != _MANIFEST_KIND:
            raise ManifestValidationError(
                "Unsupported manifest kind %r; expected %r" % (manifest.kind, _MANIFEST_KIND)
            )
        if manifest.version is not None:
            self._validate_version(manifest.version, "version")
        self._validate_requires(manifest.requires, "requires")
        for component_id, component in manifest.components.items():
            self._validate_requires(component.requires, "components.%s.requires" % component_id)

    def _validate_version(self, value: str, where: str) -> None:
        try:
            Version(value)
        except InvalidVersion as exc:
            raise ManifestValidationError("'%s' is not PEP 440 compatible: %r" % (where, value)) from exc

    def _validate_requires(self, requires: "Dict[str, str]", where: str) -> None:
        for key, specifier in requires.items():
            try:
                SpecifierSet(specifier)
            except InvalidSpecifier as exc:
                raise ManifestValidationError(
                    "%s.%s is not a valid PEP 440 specifier: %r" % (where, key, specifier)
                ) from exc


class RequirementStatus(str, Enum):
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNAVAILABLE = "unavailable"
    UNRECOGNIZED = "unrecognized"
    INVALID = "invalid"


class RequirementResult(NamedTuple):
    key: str
    required: str
    actual: "Optional[str]"
    status: RequirementStatus
    message: str


class _Resolver(NamedTuple):
    callable: "Callable[[str], Optional[str]]"
    phase: str
    display_name: str


class RequirementResolverRegistry:
    """Maps a requirement key (``python``, ``package:<name>``,
    ``runtime:<name>``, ...) to a callable that returns the installed/
    actual version string (or ``None`` if it can't be determined).

    Core only ever registers ``python`` and ``package:*`` on its
    ``default()`` registry; every ``runtime:*`` resolver is registered by
    the capability that understands that runtime (e.g. cntr registers
    ``runtime:docker-engine``/``runtime:docker-compose``).
    """

    def __init__(self):
        self._exact: "Dict[str, _Resolver]" = {}
        # dict, not list: insertion order is the registration order, and
        # replacing an existing key's value (register_prefix(replace=True))
        # keeps its original position rather than moving it to the end.
        self._prefix: "Dict[str, _Resolver]" = {}

    @classmethod
    def default(cls) -> "RequirementResolverRegistry":
        registry = cls()
        registry.register_exact("python", lambda key: platform.python_version(), phase="host",
                                display_name="Python")
        registry.register_prefix("package:", _resolve_package_version, phase="host",
                                 display_name="installed package")
        return registry

    def _validate_registration(self, key: str, resolver: "Callable[[str], Optional[str]]",
                               phase: str, display_name: "Optional[str]") -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key/prefix must be a non-empty string")
        if not callable(resolver):
            raise ValueError("resolver must be callable")
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a non-empty string")
        if display_name is not None and (not isinstance(display_name, str) or not display_name.strip()):
            raise ValueError("display_name must be a non-empty string when given")

    def register_exact(self, key: str, resolver: "Callable[[str], Optional[str]]",
                       phase: str = "host", display_name: "Optional[str]" = None,
                       replace: bool = False) -> None:
        self._validate_registration(key, resolver, phase, display_name)
        if key in self._exact and not replace:
            raise ValueError("Resolver already registered for exact key %r (pass replace=True to override)" % key)
        self._exact[key] = _Resolver(callable=resolver, phase=phase, display_name=display_name or key)

    def register_prefix(self, prefix: str, resolver: "Callable[[str], Optional[str]]",
                        phase: str = "host", display_name: "Optional[str]" = None,
                        replace: bool = False) -> None:
        """Registers a fallback resolver for every key starting with
        ``prefix``. At resolve time the *longest* matching prefix wins
        (never registration order); among equal-length matches, the
        earliest-registered one wins. Registering the exact same ``prefix``
        string twice raises unless ``replace=True``."""
        self._validate_registration(prefix, resolver, phase, display_name)
        if prefix in self._prefix and not replace:
            raise ValueError("Resolver already registered for prefix %r (pass replace=True to override)" % prefix)
        self._prefix[prefix] = _Resolver(callable=resolver, phase=phase, display_name=display_name or prefix)

    def resolve(self, key: str) -> "Optional[_Resolver]":
        if key in self._exact:
            return self._exact[key]
        best: "Optional[tuple[int, int, _Resolver]]" = None
        for order, (prefix, entry) in enumerate(self._prefix.items()):
            if not key.startswith(prefix):
                continue
            if best is None or len(prefix) > best[0] or (len(prefix) == best[0] and order < best[1]):
                best = (len(prefix), order, entry)
        return best[2] if best else None

    def check(self, requirements: "Dict[str, str]", phase: "Optional[str]" = None) -> list:
        """Evaluate every key in ``requirements`` against its registered
        resolver. A key resolved to a *different* phase than requested is
        skipped entirely (so a host-phase check never triggers a runtime
        probe); a key with no resolver at all has no phase to filter by, so
        it is always reported as ``UNRECOGNIZED`` regardless of ``phase``.
        Never raises: a resolver exception becomes an ``UNAVAILABLE``
        result rather than propagating (and its message never echoes the
        raw exception, which could carry a command/path/credential)."""
        results = []
        for key, required in requirements.items():
            entry = self.resolve(key)
            if entry is None:
                results.append(RequirementResult(
                    key=key, required=required, actual=None, status=RequirementStatus.UNRECOGNIZED,
                    message="no resolver is registered for requirement `%s`" % key,
                ))
                continue
            if phase is not None and entry.phase != phase:
                continue
            results.append(self._check_one(key, required, entry))
        return results

    def _check_one(self, key: str, required: str, entry: "_Resolver") -> "RequirementResult":
        try:
            actual = entry.callable(key)
        except Exception:  # noqa: BLE001 - never leak resolver internals into the result message
            return RequirementResult(
                key=key, required=required, actual=None, status=RequirementStatus.UNAVAILABLE,
                message="%s version could not be determined" % entry.display_name,
            )
        if actual is None:
            return RequirementResult(
                key=key, required=required, actual=None, status=RequirementStatus.UNAVAILABLE,
                message="%s version could not be determined" % entry.display_name,
            )
        try:
            specifier_set = SpecifierSet(required)
        except InvalidSpecifier:
            return RequirementResult(
                key=key, required=required, actual=actual, status=RequirementStatus.INVALID,
                message="requires.%s is not a valid PEP 440 specifier: %r" % (key, required),
            )
        try:
            satisfied = specifier_set.contains(Version(actual), prereleases=True)
        except InvalidVersion:
            return RequirementResult(
                key=key, required=required, actual=actual, status=RequirementStatus.INVALID,
                message="%s version %r is not PEP 440 comparable" % (entry.display_name, actual),
            )
        if satisfied:
            return RequirementResult(
                key=key, required=required, actual=actual, status=RequirementStatus.SATISFIED,
                message="%s %s satisfies required %s" % (entry.display_name, actual, required),
            )
        return RequirementResult(
            key=key, required=required, actual=actual, status=RequirementStatus.UNSATISFIED,
            message="%s %s does not satisfy required %s" % (entry.display_name, actual, required),
        )


def _resolve_package_version(key: str) -> "Optional[str]":
    distribution = key[len("package:"):]
    try:
        return _pkg_version(distribution)
    except PackageNotFoundError:
        return None
