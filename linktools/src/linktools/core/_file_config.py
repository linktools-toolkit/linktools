#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Linktools layered file configuration.

Two lightweight JSON file sources -- a user-level ``~/.linktools/linktools.json``
and a local-level ``<root>/.linktools.json`` -- that feed the existing
``ConfigResolver`` as plain file sources. This is a generic Linktools file
configuration source, not a Manifest protocol or an independent
configuration system: it does not require ``version``/``kind``/``schema_version``,
does not search parent directories, and only interprets two top-level keys
(``environment``, ``requires``); every other top-level field is preserved
verbatim and left for the caller to interpret via ``get_path()``/``require_path()``.
"""
import copy
import json
import os
from pathlib import Path

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

from ..errors import ConfigError, ConfigValidationError

__all__ = [
    "LinktoolsFileConfig",
    "ResolvedLinktoolsFileConfig",
    "LinktoolsFileConfigLoader",
    "merge_file_config",
    "ensure_requirement",
]

_MAX_FILE_BYTES = 1024 * 1024  # 1 MiB


def _validate_key(key, field, path):
    if not isinstance(key, str):
        raise ConfigValidationError("'%s' key must be a string in %s" % (field, path))
    if not key.strip():
        raise ConfigValidationError("'%s' key must not be empty in %s" % (field, path))
    if key != key.strip():
        raise ConfigValidationError(
            "'%s' key must not contain leading or trailing spaces in %s" % (field, path))


def _validate_file_data(data, path):
    """Static shape validation of the two Core-reserved top-level fields.

    Every other top-level field is opaque to Core: allowed, preserved, never
    interpreted or rejected here (spec §7).
    """
    if "environment" in data:
        environment = data["environment"]
        if not isinstance(environment, dict):
            raise ConfigError("'environment' must be an object in %s" % path)
        for key in environment:
            _validate_key(key, "environment", path)

    if "requires" in data:
        requires = data["requires"]
        if not isinstance(requires, dict):
            raise ConfigError("'requires' must be an object in %s" % path)
        for key, value in requires.items():
            _validate_key(key, "requires", path)
            if not isinstance(value, str) or not value.strip():
                raise ConfigValidationError("'requires.%s' must be a non-empty string in %s" % (key, path))
            if value != value.strip():
                raise ConfigValidationError(
                    "'requires.%s' must not contain leading or trailing spaces in %s" % (key, path))
            try:
                SpecifierSet(value)
            except InvalidSpecifier as exc:
                raise ConfigError(
                    "'requires.%s' is not a valid PEP 440 specifier in %s: %r" % (key, path, value)
                ) from exc


class LinktoolsFileConfig(object):
    """One parsed ``linktools.json``/``.linktools.json`` file (or an empty
    stand-in for a file that does not exist)."""

    def __init__(self, data=None, path=None):
        self._data = copy.deepcopy(data or {})
        self._path = path

    @property
    def path(self):
        return self._path

    @property
    def environment(self):
        return copy.deepcopy(self._data.get("environment", {}))

    @property
    def requires(self):
        return dict(self._data.get("requires", {}))

    def get_requirement(self, name, default=None):
        return self.requires.get(name, default)

    def get_path(self, *keys, **kwargs):
        if "default" in kwargs:
            default = kwargs.pop("default")
        else:
            default = None
        if kwargs:
            raise TypeError("get_path() got unexpected keyword argument(s): %s" % ", ".join(sorted(kwargs)))
        node = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return copy.deepcopy(node)

    def require_path(self, *keys):
        node = self._data
        for index, key in enumerate(keys):
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                raise ConfigError(
                    "config path %r is required but missing in %s"
                    % (".".join(str(k) for k in keys[:index + 1]), self._path or "<empty>")
                )
        return copy.deepcopy(node)

    def to_dict(self):
        return copy.deepcopy(self._data)

    def __repr__(self):
        return "LinktoolsFileConfig(path=%r, keys=%r)" % (self._path, sorted(self._data.keys()))


def merge_file_config(global_data, local_data):
    """Merge two plain dicts, ``local_data`` taking precedence (spec §14).

    Recurses only where *both* sides hold a dict for the same key; any other
    type mismatch (scalar, array, ``null``) is a full replacement by the
    local value. Neither input is mutated.
    """
    if isinstance(global_data, dict) and isinstance(local_data, dict):
        result = copy.deepcopy(global_data)
        for key, value in local_data.items():
            if key in result:
                result[key] = merge_file_config(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result
    return copy.deepcopy(local_data)


class ResolvedLinktoolsFileConfig(object):
    """The global file, the local file, and their merged effective view.

    All three are kept (not just the merged dict) because a requirement
    check must only ever read ``local_config`` (spec §10) while ordinary
    ``environment`` lookups use ``effective``.
    """

    def __init__(self, global_config, local_config):
        self.global_config = global_config
        self.local_config = local_config
        self.effective = LinktoolsFileConfig(
            merge_file_config(global_config.to_dict(), local_config.to_dict())
        )

    @property
    def environment(self):
        return self.effective.environment

    def get_path(self, *keys, **kwargs):
        return self.effective.get_path(*keys, **kwargs)


class LinktoolsFileConfigLoader(object):
    """Reads and merges the user-level and local-level Linktools file
    configs. Does not search parent directories; one loader call resolves
    exactly one local root (spec §5.3)."""

    global_file_name = "linktools.json"
    local_file_name = ".linktools.json"
    max_bytes = _MAX_FILE_BYTES

    def __init__(self, global_path=None):
        # For tests / embedding only (spec §5.1) -- production code should
        # not need to override this; the default is process-fixed and must
        # not depend on STORAGE_PATH (that would create a bootstrap cycle).
        self._global_path = global_path

    def get_global_path(self):
        if self._global_path is not None:
            return str(self._global_path)
        return str(Path.home().joinpath(".linktools", self.global_file_name))

    def get_local_path(self, local_root=None):
        root = local_root if local_root is not None else os.getcwd()
        return str(Path(str(root)).joinpath(self.local_file_name))

    def load_file(self, path):
        path = str(path)
        try:
            with open(path, "rb") as file:
                raw = file.read(self.max_bytes + 1)
        except FileNotFoundError:
            if os.path.lexists(path):
                # A dangling symlink (or a file that disappeared between the
                # existence check and the open) must not be treated as an
                # absent file -- that would silently discard a high-priority
                # local config the user believes is in effect (spec §18).
                raise ConfigError("config file is not readable: %s" % path)
            return LinktoolsFileConfig({}, path=path)
        except OSError as exc:
            raise ConfigError("cannot read config %s: %s" % (path, exc)) from exc

        if len(raw) > self.max_bytes:
            raise ConfigError("config file exceeds %d bytes: %s" % (self.max_bytes, path))

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError("config file must be UTF-8: %s" % path) from exc

        try:
            data = json.loads(text)
        except ValueError as exc:
            raise ConfigError("config file is not valid JSON: %s" % path) from exc

        if not isinstance(data, dict):
            raise ConfigError("config file root must be a JSON object: %s" % path)

        _validate_file_data(data, path)
        return LinktoolsFileConfig(data, path=path)

    def load(self, local_root=None):
        global_config = self.load_file(self.get_global_path())
        local_config = self.load_file(self.get_local_path(local_root))
        return ResolvedLinktoolsFileConfig(global_config, local_config)


def ensure_requirement(file_config, name, actual_version):
    """Raise ``ConfigValidationError`` if ``file_config`` declares a
    ``requires.<name>`` specifier that ``actual_version`` does not satisfy.

    ``file_config`` is expected to be a single ``LinktoolsFileConfig`` --
    callers that must isolate this to a repository's local declaration
    (spec §10, §38) pass ``resolved.local_config``, never ``resolved.effective``.
    A missing requirement means "unconstrained": this never raises.
    """
    required = file_config.get_requirement(name)
    if required is None:
        return
    try:
        specifier = SpecifierSet(required)
        version = Version(actual_version)
    except (InvalidSpecifier, InvalidVersion) as exc:
        raise ConfigValidationError(
            "invalid requirement or version for `%s`: %s" % (name, exc)
        ) from exc
    if version not in specifier:
        raise ConfigValidationError(
            "`%s` requires %s, current version is %s" % (name, required, actual_version)
        )
