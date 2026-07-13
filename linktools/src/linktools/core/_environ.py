#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The ``environ`` singleton: data/temp directory layout, logging setup,
and config access shared across every linktools command."""
import abc
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from linktools import utils, metadata
from linktools.system import get_machine, get_system
from linktools.decorator import cached_property
from linktools.types import MISSING

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import T, Config, Tools, Tool, UrlFile, PathType
    from ._paths import EnvironmentPaths
    from ._logging import LoggingManager
    from ._locks import LockManager
    from ..cache import CacheStore
    from ._config_store import ConfigStore
    from ._download import DownloadManager
    from ._profile import ProjectProfile


def _normalize_path(value: "Any") -> str:
    return os.path.abspath(os.path.expanduser(str(value)))


class ConfigDict(dict):
    """Minimal dict subclass for tool config loading (v2: replaces old _config.ConfigDict)."""

    def __init__(self, *args, **kwargs):
        self._revision = 0
        super().__init__(*args, **kwargs)

    @property
    def revision(self):
        return self._revision

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._revision += 1

    def update(self, *args, **kwargs):
        values = dict(*args, **kwargs)
        if values:
            super().update(values)
            self._revision += 1

    def clear(self):
        if self:
            super().clear()
            self._revision += 1

    def update_from_file(self, filename, load, silent=False):
        try:
            with open(filename, "rb") as f:
                obj = load(f)
        except OSError:
            if silent:
                return False
            raise
        if isinstance(obj, dict):
            self.update(obj)
        return True


class BaseEnviron(abc.ABC):
    """Base environment abstraction for config, paths, tools, and logging."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the name.

        Returns:
            str: The property value.
        """
        pass

    @property
    def version(self) -> str:
        """Return the version.

        Returns:
            str: The property value.
        """
        return NotImplemented

    @property
    def description(self) -> str:
        """Return the description.

        Returns:
            str: The property value.
        """
        return NotImplemented

    @property
    def root_path(self) -> "PathType":
        """Return the root path.

        Returns:
            PathType: The property value.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        raise NotImplementedError()

    @property
    def system(self) -> str:
        """Return the system name.

        Returns:
            str: The property value.
        """
        return get_system()

    @property
    def machine(self) -> str:
        """Return the machine architecture.

        Returns:
            str: The property value.
        """
        return get_machine()

    @property
    def debug(self) -> bool:
        """Return whether debug mode is enabled.

        Returns:
            bool: The property value.
        """
        return self.get_config("DEBUG", bool)

    @debug.setter
    def debug(self, value: bool) -> None:
        """Return whether debug mode is enabled.

        Args:
            value (bool): Value to store or process.
        """
        self.set_config("DEBUG", value)

    @cached_property
    def data_path(self) -> "Path":
        """Data path.

        Returns:
            Path: The operation result.

        A projection of ``self.paths.data`` -- ``paths`` is the single
        accessor that fixes the bootstrap filesystem layout, so accessing
        this necessarily fixes ``paths`` too.
        """
        return self.paths.data

    @cached_property
    def temp_path(self) -> "Path":
        """Temp path.

        Returns:
            Path: The operation result.

        A projection of ``self.paths.temp`` -- see ``data_path``.
        """
        return self.paths.temp

    @cached_property(lock=True)
    def paths(self) -> "EnvironmentPaths":
        """Resolved, normalised filesystem layout.

        Returns:
            EnvironmentPaths: The operation result.

        ``data``/``temp`` come from the same bootstrap resolution as the
        legacy accessors, so existing behavior is unchanged; ``cache``/
        ``config``/``logs``/``downloads`` are the new canonical locations
        consumers migrate to in later phases.
        """
        from ._paths import EnvironmentPaths

        profile = self._profile
        prefix = metadata.__name__.upper()
        environment = profile.get("environment", {})

        def resolve(key, default):
            names = (f"{prefix}_PATH", f"{prefix}_STORAGE_PATH") \
                if key == "STORAGE_PATH" else (f"{prefix}_{key}",)
            raw = environment.get(key, MISSING)
            if raw is MISSING:
                raw = next((os.environ[name] for name in names if name in os.environ), MISSING)
            if raw is MISSING:
                return default
            text = "" if raw is None else str(raw).strip()
            if not text:
                return default
            return _normalize_path(text)

        storage_path = resolve("STORAGE_PATH", _normalize_path(
            os.path.join(Path.home(), f".{metadata.__name__}")))
        data_path = resolve("DATA_PATH", _normalize_path(os.path.join(storage_path, "data")))
        temp_path = resolve("TEMP_PATH", _normalize_path(os.path.join(storage_path, "temp")))

        return EnvironmentPaths(
            root=self.root_path,
            storage=storage_path,
            data=data_path,
            temp=temp_path,
        )

    @cached_property(lock=True)
    def locks(self) -> "LockManager":
        """Unified process/file lock manager.

        Returns:
            LockManager: The operation result.
        """
        from ._locks import LockManager

        return LockManager(self.paths.cache / "locks")

    @cached_property(lock=True)
    def cache(self) -> "CacheStore":
        """Transactional local cache.

        Returns:
            CacheStore: The operation result.

        A single SQLite store shared across the process; use
        ``environ.cache.namespace(name)`` for isolated key spaces. Per-thread
        connections are managed inside the store.
        """
        from ..cache import CacheStore

        self.paths.ensure_cache()
        return CacheStore(self.paths.cache / "cache.db")

    @cached_property(lock=True)
    def config_store(self) -> "ConfigStore":
        """Persistent, user-editable JSON store.

        Returns:
            ConfigStore: The operation result.

        The proper home for persistent user state (e.g. cntr's
        INSTALLED_CONTAINERS) that must NOT live in the cache. Distinct file and
        lifecycle from ``cache``; written atomically under a process lock.
        """
        from ._config_store import ConfigStore

        self.paths.ensure_config()
        return ConfigStore(self.paths.config / "settings.json", lock_manager=self.locks)

    @cached_property(lock=True)
    def downloads(self) -> "DownloadManager":
        """Unified download manager.

        Returns:
            DownloadManager: The operation result.

        Locks via ``self.locks``, stores resume metadata in ``self.cache``;
        consumers migrate from the legacy UrlFile (core/_url.py) to this.
        """
        from ._download import DownloadManager

        return DownloadManager(self)

    def subprocess_env(self, include_tools=True, overrides=None):
        """Build a subprocess environment dict.

        Returns a fresh mapping (never mutates the process ``os.environ``): the
        managed-tools stub dir is prepended to PATH so tools resolve without the
        global PATH mutation the legacy ``_create_tools`` did, then ``overrides``
        are applied.
        """
        env = dict(os.environ)
        if include_tools:
            try:
                stub = str(self.tools.stub_path)
                if stub:
                    rest = [p for p in env.get("PATH", "").split(os.pathsep)
                            if p and p != stub]
                    env["PATH"] = os.pathsep.join([stub] + rest)
            except Exception:
                pass
        if overrides:
            env.update(overrides)
        return env

    def get_path(self, *paths: str) -> "Path":
        """Return the path.

        Args:
            paths (str): The paths value.

        Returns:
            Path: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        if self.root_path == NotImplemented:
            raise RuntimeError("root_path not implemented")
        return utils.join_path(self.root_path, *paths)

    def get_data_path(self, *paths: str, create_parent: bool = False) -> "Path":
        """Return the data path.

        Args:
            paths (str): The paths value.
            create_parent (bool): The create_parent value.

        Returns:
            Path: The operation result.
        """
        path = utils.join_path(self.data_path, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_temp_path(self, *paths: str, create_parent: bool = False) -> "Path":
        """Return the temp path.

        Args:
            paths (str): The paths value.
            create_parent (bool): The create_parent value.

        Returns:
            Path: The operation result.
        """
        path = utils.join_path(self.temp_path, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def clean_temp_files(self, *paths: str, expire_days: int = 7) -> None:
        """Remove expired temporary files.

        Args:
            paths (str): The paths value.
            expire_days (int): The expire_days value.
        """
        current_time = time.time()
        target_time = current_time - expire_days * 24 * 60 * 60

        temp_path = self.get_temp_path(*paths)
        for root, dirs, files in os.walk(temp_path, topdown=False):
            for name in files:
                path = os.path.join(root, name)
                last_time = max(
                    os.path.getatime(path),
                    os.path.getctime(path),
                    os.path.getmtime(path),
                )
                if last_time < target_time:
                    self.logger.info(f"Remove expired temp file: {path}")
                    os.remove(path)
            for name in dirs:
                path = os.path.join(root, name)
                if os.path.exists(path) and not os.listdir(path):
                    last_time = max(
                        os.path.getatime(path),
                        os.path.getctime(path),
                        os.path.getmtime(path),
                    )
                    if last_time < target_time:
                        self.logger.info(f"Remove empty temp directory: {path}")
                        shutil.rmtree(path, ignore_errors=True)

    @cached_property
    def logger(self) -> "logging.Logger":
        """Logger.

        Returns:
            logging.Logger: The operation result.
        """
        return self.get_logger()

    @cached_property(lock=True)
    def logging(self) -> "LoggingManager":
        """The :class:`LoggingManager` owning redaction, context and levels.

        Returns:
            LoggingManager: The operation result.

        Constructing it is side-effect free; redaction is installed lazily on
        the first ``get_logger``/``bootstrap``/``configure`` call.
        """
        from ._logging import LoggingManager

        return LoggingManager(self)

    def get_logger(self, name: str = None) -> "logging.Logger":
        """Return a named logger with redaction active.

        Avoids double-prefixing: if ``name`` already starts with the environment
        name (e.g. ``linktools.ssh``), it is used as-is.
        """
        if name and (name == self.name or name.startswith(self.name + ".")):
            full = name
        elif name:
            full = "%s.%s" % (self.name, name)
        else:
            full = self.name
        return self.logging.get_logger(full)

    @cached_property(lock=True)
    def global_config(self) -> "ConfigDict":
        """Global config values from the profile plus runtime flags.

        Returns:
            ConfigDict: The operation result.
        """
        return ConfigDict(DEBUG=False)

    @cached_property(lock=True)
    def _profile(self) -> "ProjectProfile":
        """Merged global and local profile for this process's own root.

        This profile is shared by bootstrap path resolution and global config.
        """
        from ._profile import ProjectProfile

        return ProjectProfile(ProjectProfile.global_path(), ProjectProfile.local_path())

    def _create_config(self) -> "Config":
        """Build the process-wide, ConfigSchema-backed main Config.

        Source precedence: EnvironmentSource > RuntimeOverrideSource
        > PersistentSource > global-config > DefaultSource.
        """
        return self.build_config("main", self.name.upper() + "_")

    @cached_property(lock=True)
    def config(self) -> "Config":
        """The process-wide, ConfigSchema-backed main Config."""
        return self._create_config()

    def build_config(self, namespace: str, env_prefix: str = "") -> "Config":
        """Build a Config for ``namespace``.

        Source precedence: EnvironmentSource > RuntimeOverrideSource >
        PersistentSource > global-config > DefaultSource.
        """
        from ._config import (
            ConfigSchema, EnvironmentSource, RuntimeOverrideSource, PersistentSource,
            Config as NewConfig, DictSource, DefaultSource,
        )

        schema = ConfigSchema()
        profile = self._profile
        return NewConfig(
            self,
            schema,
            sources=[
                EnvironmentSource(
                    (profile.get("config", {}), ""),
                    (profile.get("environment", {}), env_prefix),
                    (os.environ, env_prefix),
                ),
                RuntimeOverrideSource(),
                PersistentSource(self.config_store, namespace),
                DictSource(self.global_config, name="global-config"),
                DefaultSource(schema),
            ],
        )

    def get_config(self, key: str, type: "type[T]" = None, default: "Any" = MISSING) -> "T":
        """Return a configuration value.

        Args:
            key (str): Configuration or item key.
            type (Type[T]): Target type used to cast the value.
            default (Any): Value returned when no explicit value is available.

        Returns:
            T: The operation result.
        """
        return self.config.get(key=key, type=type, default=default)

    def require_config(self, key: str, type: "type[T]" = None) -> "T":
        """Return a must-exist configuration value; raise if it is missing."""
        return self.config.require(key=key, type=type)

    def set_config(self, key: str, value: "Any") -> None:
        """Set a configuration value.

        Args:
            key (str): Configuration or item key.
            value (Any): Value to store or process.
        """
        self.config.set(key, value)

    def close(self):
        """Close all owned resources.

        Idempotent. Closes cache connections, logging handlers, and download
        tasks owned by this Environment. Does NOT close other Environment's
        resources or the root logger.
        """
        # Cache (SQLite connections are per-thread; close this thread's).
        cache = getattr(self, "_cache", None)
        if cache is not None:
            cache.close()
        # Config store (no persistent connection to close, but flush is atomic).
        # Logging (unregister redactor from global factory).
        logging_mgr = getattr(self, "_logging", None)
        if logging_mgr is not None:
            logging_mgr.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _create_tools(self) -> "Tools":
        from ._tools import Tools
        # ConfigDict is defined above (inlined)

        config = ConfigDict()

        develop_path = self.get_path("assets", "develop", "tools.yml")
        data_path = self.get_data_path("tools", "tools.json")
        asset_path = self.get_path("assets", "tools.json")

        if os.path.exists(data_path):
            config.update_from_file(data_path, json.load)
        elif not metadata.__develop__ or not os.path.exists(develop_path):
            config.update_from_file(asset_path, json.load)
        else:
            import yaml
            config.update_from_file(develop_path, yaml.safe_load)

        tools = Tools(self, config)
        # do NOT mutate os.environ["PATH"]. Subprocesses that need the
        # tools stub resolve it via env.subprocess_env() (Tool.popen, ToolRunner).
        return tools

    @cached_property(lock=True)
    def tools(self) -> "Tools":
        """Tools.

        Returns:
            Tools: The operation result.
        """
        return self._create_tools()

    def get_tool(self, name: str, **kwargs) -> "Tool":
        """Return a configured tool by name.

        Args:
            name (str): Name to resolve.
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Tool: The operation result.
        """
        tool = self.tools[name]
        if len(kwargs) != 0:
            tool = tool.copy(**kwargs)
        return tool

    def get_url_file(self, url: "PathType") -> "UrlFile":
        """Return a URL file wrapper for a URL.

        Args:
            url (PathType): URL to process.

        Returns:
            UrlFile: The operation result.
        """
        from ._download import HttpFile, LocalFile

        if not isinstance(url, str):
            url = str(url)

        if url.startswith("http://") or url.startswith("https://"):  # noqa
            return HttpFile(self, url)
        elif url.startswith("file://"):
            return LocalFile(self, url[len("file://"):])

        return LocalFile(self, url)


class Environ(BaseEnviron):
    """Default environment implementation for the linktools package."""

    @property
    def name(self) -> str:
        """Return the name.

        Returns:
            str: The property value.
        """
        return metadata.__name__

    @property
    def version(self) -> str:
        """Return the version.

        Returns:
            str: The property value.
        """
        return metadata.__version__

    @property
    def description(self) -> str:
        """Return the description.

        Returns:
            str: The property value.
        """
        return metadata.__description__

    @cached_property
    def root_path(self) -> "Path":
        """Return the root directory for the current package.

        Returns:
            Path: The operation result.
        """
        return Path(os.path.dirname(os.path.dirname(__file__)))

    def _create_config(self):
        config = super()._create_config()

        # Initialize download-related defaults on the new Config.
        config.update_defaults(
            DEFAULT_USER_AGENT=
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0.0.0 "
            "Safari/537.36",
            DEFAULT_WAN_IP_URL="https://ifconfig.me/ip"  # noqa
        )

        return config


environ = Environ()
