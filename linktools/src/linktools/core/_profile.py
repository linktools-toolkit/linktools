#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Layered project profiles (``linktools.json`` / ``.linktools.json``)."""
import copy
import json
import os
from pathlib import Path

from ..errors import ConfigError

__all__ = [
    "ProjectProfile",
]

_MAX_FILE_BYTES = 1024 * 1024


class ProjectProfile(object):
    """One profile file, or an ordered merge of multiple profile files.

    ``ProjectProfile(path)`` reads one file. Multiple paths are merged from
    left to right, with earlier files taking precedence. A dict input remains
    an internal construction form for already parsed data.
    """

    max_bytes = _MAX_FILE_BYTES

    GLOBAL_FILE_NAME = "linktools.json"
    LOCAL_FILE_NAME = ".linktools.json"

    @classmethod
    def global_path(cls) -> str:
        """The user-level profile path: ``~/.linktools/linktools.json``."""
        return str(Path.home() / ".linktools" / cls.GLOBAL_FILE_NAME)

    @classmethod
    def local_path(cls, root=None) -> str:
        """The local-level profile path under ``root`` (the process CWD
        when ``root`` is ``None``): ``<root>/.linktools.json``."""
        return str(Path(str(root) if root is not None else os.getcwd()) / cls.LOCAL_FILE_NAME)

    def __init__(self, *paths):
        if not paths or paths[0] is None or isinstance(paths[0], dict):
            self._data = copy.deepcopy((paths[0] if paths else {}) or {})
            return

        if not all(isinstance(item, (str, bytes, os.PathLike)) for item in paths):
            raise TypeError("ProjectProfile expects profile paths")

        merged_data = {}
        for profile_path in paths:
            profile = self._load_file(profile_path)
            merged_data = self._merge(profile._data, merged_data)
        self._data = merged_data

    @staticmethod
    def _merge(base, overlay):
        """Merge two profile dictionaries, with ``overlay`` taking precedence."""
        if isinstance(base, dict) and isinstance(overlay, dict):
            result = copy.deepcopy(base)
            for key, value in overlay.items():
                result[key] = ProjectProfile._merge(result[key], value) \
                    if key in result else copy.deepcopy(value)
            return result
        return copy.deepcopy(overlay)

    @classmethod
    def _load_file(cls, path):
        path = str(path)
        try:
            with open(path, "rb") as file:
                raw = file.read(cls.max_bytes + 1)
        except FileNotFoundError:
            if os.path.lexists(path):
                raise ConfigError("config file is not readable: %s" % path)
            return cls({})
        except OSError as exc:
            raise ConfigError("cannot read config %s: %s" % (path, exc)) from exc

        if len(raw) > cls.max_bytes:
            raise ConfigError("config file exceeds %d bytes: %s" % (cls.max_bytes, path))
        try:
            text = raw.decode("utf-8")
            data = json.loads(text)
        except UnicodeDecodeError as exc:
            raise ConfigError("config file must be UTF-8: %s" % path) from exc
        except ValueError as exc:
            raise ConfigError("config file is not valid JSON: %s" % path) from exc

        if not isinstance(data, dict):
            raise ConfigError("config file root must be a JSON object: %s" % path)
        return cls(data)

    def get(self, key, default=None):
        """Return a top-level profile value, or ``default`` if absent."""
        return copy.deepcopy(self._data.get(key, default))

    def get_path(self, *keys, **kwargs):
        default = kwargs.pop("default", None)
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
                    "config path %r is required but missing"
                    % ".".join(str(k) for k in keys[:index + 1])
                )
        return copy.deepcopy(node)

    def to_dict(self):
        return copy.deepcopy(self._data)

    def __repr__(self):
        return "ProjectProfile(keys=%r)" % sorted(self._data.keys())
