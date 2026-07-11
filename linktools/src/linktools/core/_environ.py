#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : environment.py
@time    : 2020/03/01
@site    :
@software: PyCharm

              ,----------------,              ,---------,
         ,-----------------------,          ,"        ,"|
       ,"                      ,"|        ,"        ,"  |
      +-----------------------+  |      ,"        ,"    |
      |  .-----------------.  |  |     +---------+      |
      |  |                 |  |  |     | -==----'|      |
      |  | $ sudo rm -rf / |  |  |     |         |      |
      |  |                 |  |  |/----|`---=    |      |
      |  |                 |  |  |   ,/|==== ooo |      ;
      |  |                 |  |  |  // |(((( [33]|    ,"
      |  `-----------------'  |," .;'| |((((     |  ,"
      +-----------------------+  ;;  | |         |,"
         /_)______________(_/  //'   | +---------+
    ___________________________/___  `,
   /  oooooooooooooooo  .o.  oooo /,   `,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,``--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""
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
from linktools.decorator import cached_property, cached_classproperty
from linktools.types import MISSING
from ..errors import ConfigValidationError

if TYPE_CHECKING:
    from typing import Any
    from linktools.types import T, Config, Tools, Tool, UrlFile, PathType
    from ._paths import EnvironmentPaths
    from ._logging import LoggingManager
    from ._locks import LockManager
    from ..cache import CacheStore
    from ._config import ConfigSchema, ConfigSource
    from ._config_store import ConfigStore
    from ._download import DownloadManager
    from ._file_config import ResolvedLinktoolsFileConfig

# STORAGE_PATH/DATA_PATH/TEMP_PATH (spec Part IV): everything else
# (ConfigStore, cache, logs, downloads) is derived from these, so once any of
# them has actually been used to fix a path, changing the underlying value
# out from under an already-running Environ instance would leave stale
# derived paths. They are never stored in Config at all (see get_config()/
# set_config()/_bootstrap_path()) -- reads go straight to self.paths/
# data_path/temp_path, and writes are always rejected.
_BOOTSTRAP_KEYS = frozenset({"STORAGE_PATH", "DATA_PATH", "TEMP_PATH"})


def _normalize_path(value: "Any") -> str:
    return os.path.abspath(os.path.expanduser(str(value)))


def _resolve_bootstrap_paths(file_environment: dict) -> "tuple[str, str, str]":
    """Resolve STORAGE_PATH/DATA_PATH/TEMP_PATH (spec Part IV §26).

    Priority: OS environment variable > local ``.linktools.json`` (cwd) >
    user ``~/.linktools/linktools.json`` > builtin default. All three are
    normalized to an absolute path regardless of source -- a relative
    STORAGE_PATH (e.g. the spec's own third-party-repo example, "./storage")
    must resolve to the same absolute path every downstream consumer
    (EnvironmentPaths, ConfigStore, get_data_path) assumes, or the first
    ConfigStore access crashes on a relative/absolute path mismatch.
    """
    prefix = metadata.__name__.upper()

    storage_path = (
        os.environ.get(f"{prefix}_PATH")
        or os.environ.get(f"{prefix}_STORAGE_PATH")
        or file_environment.get("STORAGE_PATH")
        or os.path.join(Path.home(), f".{metadata.__name__}")
    )
    storage_path = _normalize_path(storage_path)

    data_path = os.environ.get(f"{prefix}_DATA_PATH") or file_environment.get("DATA_PATH") \
        or os.path.join(storage_path, "data")
    temp_path = os.environ.get(f"{prefix}_TEMP_PATH") or file_environment.get("TEMP_PATH") \
        or os.path.join(storage_path, "temp")

    return storage_path, _normalize_path(data_path), _normalize_path(temp_path)


class ConfigDict(dict):
    """Minimal dict subclass for tool config loading (v2: replaces old _config.ConfigDict)."""
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
        """Resolved, normalised filesystem layout (spec §5.3).

        Returns:
            EnvironmentPaths: The operation result.

        ``data``/``temp`` come from the same global config as the legacy
        accessors, so existing behavior is unchanged; ``cache``/``config``/
        ``logs``/``downloads`` are the new canonical locations consumers migrate
        to in later phases.
        """
        from ._paths import EnvironmentPaths

        cfg = self.global_config
        return EnvironmentPaths(
            root=self.root_path,
            storage=cfg["STORAGE_PATH"],
            data=cfg["DATA_PATH"],
            temp=cfg["TEMP_PATH"],
        )

    @cached_property(lock=True)
    def locks(self) -> "LockManager":
        """Unified process/file lock manager (spec §7.11 CAC-009).

        Returns:
            LockManager: The operation result.
        """
        from ._locks import LockManager

        return LockManager(self.paths.cache / "locks")

    @cached_property(lock=True)
    def cache(self) -> "CacheStore":
        """Transactional local cache (spec §7, §5.1).

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
        """Persistent, user-editable JSON store (spec §8.5 CFG-005).

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
        """Unified download manager (spec §9, §5.1).

        Returns:
            DownloadManager: The operation result.

        Locks via ``self.locks``, stores resume metadata in ``self.cache``;
        consumers migrate from the legacy UrlFile (core/_url.py) to this.
        """
        from ._download import DownloadManager

        return DownloadManager(self)

    def subprocess_env(self, include_tools=True, overrides=None):
        """Build a subprocess environment dict (spec §5.1, §10.11).

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
        the first ``get_logger``/``bootstrap``/``configure`` call (spec §5.4).
        """
        from ._logging import LoggingManager

        return LoggingManager(self)

    def get_logger(self, name: str = None) -> "logging.Logger":
        """Return a named logger with redaction active (spec §3.2/§5.4, v2 §4.3).

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

    @cached_classproperty(lock=True)
    def global_config(self) -> "ConfigDict":
        """Build the global configuration dictionary.

        Returns:
            ConfigDict: The operation result.

        Bootstrap priority (spec Part IV §26): OS environment variable >
        local ``.linktools.json`` (cwd) > user ``~/.linktools/linktools.json``
        > builtin default. Deliberately excludes PersistentSource (its own
        path depends on STORAGE_PATH -- reading it here would be circular)
        and any Prompt/Lazy provider. A corrupt/oversized bootstrap file
        raises (via LinktoolsFileConfigLoader) rather than being silently
        treated as absent -- see spec §18.
        """
        # ConfigDict is defined above (inlined)
        from ._file_config import LinktoolsFileConfigLoader

        file_environment = LinktoolsFileConfigLoader().load().environment
        storage_path, data_path, temp_path = _resolve_bootstrap_paths(file_environment)

        return ConfigDict(
            DEBUG=False,
            STORAGE_PATH=storage_path,
            DATA_PATH=data_path,
            TEMP_PATH=temp_path,
        )

    def load_file_config(self, local_root: "PathType | None" = None) -> "ResolvedLinktoolsFileConfig":
        """Load and merge the user-level and local-level linktools.json
        (spec Part I-II). ``local_root`` defaults to the current working
        directory; a third-party repository context (spec Part V) passes its
        own root instead.
        """
        from ._file_config import LinktoolsFileConfigLoader

        return LinktoolsFileConfigLoader().load(local_root=local_root)

    def _build_file_sources(self, local_root: "PathType | None" = None) -> "tuple[ConfigSource, ConfigSource]":
        """Build the local-file/global-file FileSource pair for a Config's
        source chain (spec §21).

        Both sources reload atomically (spec §24): the local source's
        ``reload_fn`` re-reads *both* files as a single
        ``load_file_config()`` call and pushes the global half straight into
        ``global_source`` (via the public ``replace()``, never touching
        ``_data`` directly) before returning its own half, so either both
        move to a freshly-read, mutually consistent pair, or -- if either
        file is now missing/corrupt -- neither source's data changes at all
        (the exception propagates out of ``reload_fn`` before either is
        touched).

        Each source's ``base_path`` is the directory its backing
        file lives in, so a ``cast="path"`` field's relative value resolves
        against that config file's own directory, not the process CWD:
        global-file -> ``~/.linktools/``; local-file -> ``local_root`` (or
        the CWD when ``local_root`` is None, matching
        ``LinktoolsFileConfigLoader.get_local_path``).
        """
        from ._config import FileSource
        from ._file_config import LinktoolsFileConfigLoader

        loader = LinktoolsFileConfigLoader()
        global_base_path = os.path.dirname(loader.get_global_path())
        local_base_path = str(local_root) if local_root is not None else os.getcwd()

        resolved = self.load_file_config(local_root=local_root)
        local_source = FileSource(resolved.local_config.environment, name="local-file",
                                  base_path=local_base_path)
        global_source = FileSource(resolved.global_config.environment, name="global-file",
                                   base_path=global_base_path)

        def _reload_local() -> "tuple[dict, str]":
            fresh = self.load_file_config(local_root=local_root)
            global_source.replace(fresh.global_config.environment, base_path=global_base_path)
            return fresh.local_config.environment, local_base_path

        local_source._reload_fn = _reload_local
        global_source._reload_fn = lambda: (global_source._data, global_source.base_path)

        return local_source, global_source

    def _bootstrap_path(self, key: str) -> str:
        """Return the already-fixed value of a bootstrap key (spec §26-27).

        STORAGE_PATH/DATA_PATH/TEMP_PATH are deliberately NOT stored in
        ``Config`` at all (no field, no source ever resolves them) -- they
        live solely on ``self.paths``/``self.data_path``/``self.temp_path``,
        computed once from ``global_config`` (env var / linktools.json /
        builtin default). This is what makes them immune to Config-level
        state by construction: a stale value in PersistentSource, a runtime
        override, or a reload() picking up a changed file can never disagree
        with ``self.paths.storage`` -- nothing ever asks Config's opinion of
        them in the first place, so there is no drift to detect.
        """
        return str({
            "STORAGE_PATH": self.paths.storage,
            "DATA_PATH": self.data_path,
            "TEMP_PATH": self.temp_path,
        }[key])

    def _create_config(self) -> "Config":
        """Build the new Config (ConfigSchema-backed, v2 §3 main path).

        Sources (spec §19 precedence): EnvironmentSource > RuntimeOverrideSource
        > PersistentSource > local-file > global-file > DefaultSource.
        STORAGE_PATH/DATA_PATH/TEMP_PATH are intentionally absent from this
        schema -- see ``_bootstrap_path``/``get_config``/``set_config``.
        """
        from ._config import (
            Config as NewConfig, ConfigSchema,
            EnvironmentSource, RuntimeOverrideSource,
            PersistentSource, DefaultSource,
        )

        schema = ConfigSchema(allow_unknown=True)  # dynamic keys (DEBUG etc.)
        prefix = self.name.upper() + "_"
        local_source, global_source = self._build_file_sources()
        config = NewConfig(
            self,
            schema,
            sources=[
                EnvironmentSource(prefix),
                RuntimeOverrideSource(),
                PersistentSource(self.config_store, "main"),
                local_source,
                global_source,
                DefaultSource(schema),
            ],
        )
        config.update_defaults(DEBUG=False)
        return config

    @cached_property(lock=True)
    def config(self) -> "Config":
        """Config (v2 §3: new ConfigSchema-backed main path)."""
        return self._create_config()

    def wrap_config(self, namespace=MISSING, env_prefix=MISSING, local_root: "PathType | None" = None):
        """Return a scoped Config (v2 §3: new Config, not ConfigWrapper).

        Each call returns a fresh Config with its own schema AND its own
        fresh Environment/RuntimeOverride/Persistent sources, so sub-managers
        (cntr) can define their own fields independently. ``local_root``
        (spec Part V) lets a caller point the local-file layer at a
        third-party repository root instead of the current working
        directory -- the default used for the process-wide ``config``/cwd
        case.

        A caller that needs *multiple sibling* Config objects to share the
        same Environment/RuntimeOverride/Persistent state (so a runtime
        override or a persisted value uniformly overrides every sibling,
        spec §33/§71) must not call this repeatedly -- use
        ``shared_config_sources()`` + ``build_config()`` instead, each with
        its own ``local_root``.
        """
        from ._config import ConfigSchema

        schema = ConfigSchema(allow_unknown=True)  # Tools/cntr dynamic keys
        prefix = (env_prefix if env_prefix is not MISSING else "")
        ns = namespace if namespace is not MISSING else "main"
        return self.build_config(schema, self.shared_config_sources(ns, prefix), local_root=local_root)

    def shared_config_sources(self, namespace: str, env_prefix: str = "") -> "tuple[ConfigSource, ConfigSource, ConfigSource]":
        """Build one (Environment, RuntimeOverride, Persistent) source triple
        for ``namespace`` (spec §33). Pass the SAME returned tuple to
        multiple ``build_config()`` calls (varying only ``local_root``) so
        every sibling Config shares one process-wide runtime-override state
        and reads/writes the same persisted namespace -- only the local-file
        layer is allowed to differ per sibling.
        """
        from ._config import EnvironmentSource, RuntimeOverrideSource, PersistentSource

        return (
            EnvironmentSource(env_prefix),
            RuntimeOverrideSource(),
            PersistentSource(self.config_store, namespace),
        )

    def build_config(self, schema: "ConfigSchema", shared_sources: "tuple[ConfigSource, ConfigSource, ConfigSource]",
                      local_root: "PathType | None" = None) -> "Config":
        """Build a Config from ``schema``, a ``shared_config_sources()``
        triple, and this instance's own fresh local-file/global-file
        FileSource pair for ``local_root``.
        """
        from ._config import Config as NewConfig, DefaultSource

        env_source, runtime_source, persistent_source = shared_sources
        local_source, global_source = self._build_file_sources(local_root=local_root)
        return NewConfig(
            self,
            schema,
            sources=[
                env_source,
                runtime_source,
                persistent_source,
                local_source,
                global_source,
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

        STORAGE_PATH/DATA_PATH/TEMP_PATH bypass ``self.config`` entirely (see
        ``_bootstrap_path``) -- they are always present, so ``default`` is
        never used for them.
        """
        if key in _BOOTSTRAP_KEYS:
            value = self._bootstrap_path(key)
            return type(value) if type is not None else value
        return self.config.get(key=key, type=type, default=default)

    def require_config(self, key: str, type: "type[T]" = None) -> "T":
        """Return a must-exist configuration value; raise if it is missing."""
        if key in _BOOTSTRAP_KEYS:
            return self.get_config(key, type=type)
        return self.config.require(key=key, type=type)

    def set_config(self, key: str, value: "Any") -> None:
        """Set a configuration value.

        Args:
            key (str): Configuration or item key.
            value (Any): Value to store or process.

        STORAGE_PATH/DATA_PATH/TEMP_PATH are fixed at bootstrap and are never
        settable at runtime (spec §29) -- edit the linktools.json file(s) or
        the OS environment variable and restart instead.
        """
        if key in _BOOTSTRAP_KEYS:
            raise ConfigValidationError(
                "%r cannot be changed at runtime; edit the linktools.json "
                "file(s) or the OS environment variable and restart instead" % (key,)
            )
        self.config.set(key, value)

    def close(self):
        """Close all owned resources (v4 §10.3).

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
