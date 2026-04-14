#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@author  : Hu Ji
@file    : _config.py
@time    : 2023/05/20
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
import configparser
import errno
import json
import os
import shutil
import threading
from collections import ChainMap
from pathlib import Path
from types import ModuleType
from typing import \
    TYPE_CHECKING, Type, Optional, Generator, \
    Any, Tuple, IO, Mapping, Union, List, Dict, Callable

from linktools import utils
from linktools.metadata import __missing__
from linktools.rich import choose, prompt, confirm
from linktools.types import PathType, get_args, ConfigError, FileCache

if TYPE_CHECKING:
    from typing import Self, Literal
    from ..types import T
    from ._environ import BaseEnviron

    ConfigLiteralType = Literal["path", "json"]
    ConfigType = Union[ConfigLiteralType, Callable[[Any], T], None]
    ConfigTypeMap = Dict[ConfigType, Callable[[Any], T]]

SUPPRESS = object()


def is_type(obj: Any) -> bool:
    """Return whether a value already matches the requested config type.

    Args:
        obj (Any): Object to inspect or convert.

    Returns:
        bool: The operation result.
    """
    return isinstance(obj, type)


def cast_bool(obj: Any) -> bool:
    """Cast a config value to bool.

    Args:
        obj (Any): Object to inspect or convert.

    Returns:
        bool: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        data = obj.lower()
        if data in ("true", "yes", "y", "on", "1"):
            return True
        elif data in ("false", "no", "n", "off", "0"):
            return False
        raise TypeError(f"str '{obj}' cannot be converted to type bool")
    return bool(obj)


def cast_str(obj: Any) -> str:
    """Cast a config value to str.

    Args:
        obj (Any): Object to inspect or convert.

    Returns:
        str: The operation result.
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (Tuple, List, Dict)):
        return json.dumps(obj)
    if obj is None:
        return ""
    return str(obj)


def cast_path(obj: Any) -> Path:
    """Cast a config value to an expanded filesystem path.

    Args:
        obj (Any): Object to inspect or convert.

    Returns:
        Path: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    if isinstance(obj, get_args(PathType)):
        return Path(
            os.path.expanduser(
                str(obj)  # support Proxy object
            )
        )
    raise TypeError(f"{type(obj)} cannot be converted to path")


def cast_json(obj: Any) -> Union[List, Dict]:
    """Cast a config value to JSON-compatible data.

    Args:
        obj (Any): Object to inspect or convert.

    Returns:
        Union[List, Dict]: The operation result.

    Raises:
        Exception: Propagates errors raised while completing the operation.
    """
    if isinstance(obj, str):
        return json.loads(obj)
    if isinstance(obj, (Tuple, List, Dict)):
        return obj
    raise TypeError(f"{type(obj)} cannot be converted to json")


CONFIG_TYPES: "ConfigTypeMap" = dict({
    bool: cast_bool,
    str: cast_str,
    "path": cast_path,
    "json": cast_json,
})


class ConfigProperty(metaclass=abc.ABCMeta):

    """Descriptor for reading and writing typed config values."""
    def __init__(self, *, type: "ConfigType" = None, default: Any = __missing__):
        self._type = type
        self._default = default
        self._tail = None
        if default is __missing__:
            self._tail = self
        elif isinstance(default, ConfigProperty):
            self._tail = default._tail

    @property
    def type(self) -> "ConfigType":
        """Type.

        Returns:
            ConfigType: The property value.
        """
        return self._type

    @property
    def default(self) -> Any:
        """Default.

        Returns:
            Any: The property value.
        """
        return self._default

    @abc.abstractmethod
    def get(self, config: "Config", key: str, *, type: "ConfigType", default: Any, **kwargs) -> Any:
        """Return a resolved config value.

        Args:
            config (Config): The config value.
            key (str): Configuration or item key.
            type (ConfigType): Target type used to cast the value.
            default (Any): Value returned when no explicit value is available.
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Any: The operation result.
        """
        pass

    def set_default(self, value: Any, ignore_errors: bool = False) -> "Self":
        """Set the default value for this config property chain.

        Args:
            value (Any): Value to store or process.
            ignore_errors (bool): Whether command errors should be suppressed.

        Returns:
            Self: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        if self._tail is None:
            if not ignore_errors:
                raise ValueError("config default value has been set, cannot use \"|\" operator")
            return self

        # update tail if necessary
        if self._tail._default is not __missing__:
            tail, default = self._tail, self._tail._default
            while default is not __missing__:
                if not isinstance(default, ConfigProperty):
                    tail = None
                    break
                tail, default = default, tail._default
            node = self
            while isinstance(node, ConfigProperty):
                node._tail = tail
                node = node._default
            if tail is None:
                if not ignore_errors:
                    raise ValueError("config default value has been set, cannot use \"|\" operator")
                return self

        if self._tail is self:
            self._default = value
            self._tail = value \
                if isinstance(value, ConfigProperty) and value._default is __missing__ \
                else None
        else:
            self._tail.set_default(value)
            self._tail = self._tail._tail

        return self

    def __or__(self, other: Any) -> "Self":
        return self.set_default(other, ignore_errors=False)


class LazyConfigProperty(ConfigProperty, metaclass=abc.ABCMeta):

    """Config property whose default value is computed lazily."""
    def get(self, config: "Config", key: str, *, type: "ConfigType", default: Any, **kwargs) -> Any:
        """Resolve a lazy config value.

        Args:
            config (Config): The config value.
            key (str): Configuration or item key.
            type (ConfigType): Target type used to cast the value.
            default (Any): Value returned when no explicit value is available.
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Any: The operation result.
        """
        result = self.load(config, key, type=type, default=default, **kwargs)
        if isinstance(result, ConfigProperty):
            result = result.get(config, key, type=type, default=default, **kwargs)
        return result

    @abc.abstractmethod
    def load(self, config: "Config", key: str, *, type: "ConfigType", default: Any, **kwargs) -> Any:
        """Load the lazy config value.

        Args:
            config (Config): The config value.
            key (str): Configuration or item key.
            type (ConfigType): Target type used to cast the value.
            default (Any): Value returned when no explicit value is available.
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Any: The operation result.
        """
        pass


class CacheConfigProperty(ConfigProperty, metaclass=abc.ABCMeta):

    """Config property that can persist prompted values."""
    def __init__(self, *, type: "ConfigType" = None, default: Any, cached: bool = __missing__):
        super().__init__(type=type, default=default)
        self._data = __missing__
        self._cached = cached

    def get(self, config: "Config", key: str, type: "ConfigType", default: Any, **kwargs) -> Any:
        """Resolve a cached config value.

        Args:
            config (Config): The config value.
            key (str): Configuration or item key.
            type (ConfigType): Target type used to cast the value.
            default (Any): Value returned when no explicit value is available.
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Any: The operation result.
        """
        if self._data is not __missing__:
            return self._data

        type = type or self._type
        if self._cached:
            # load cache from config file
            parser = ConfigCacheParser(config.cache.path, config.cache.namespace)
            cache = default
            if cache is __missing__:
                cache = parser.get(key, __missing__)

            # load config value
            result = self.load(config, key, type=type, cache=cache, **kwargs)
            if isinstance(result, ConfigProperty):
                result = result.get(config, key, type=type, default=default, **kwargs)
            elif type is not None:
                result = config.cast(result, type)

            # update cache to config file
            parser.set(key, cast_str(result))
            parser.dump()

        else:
            result = self.load(config, key, type=type, cache=default, **kwargs)
            if isinstance(result, ConfigProperty):
                result = result.get(config, key, type=type, default=default, **kwargs)
            elif type is not None:
                result = config.cast(result, type)

        self._data = result
        return result

    @abc.abstractmethod
    def load(self, config: "Config", key: str, *, type: "ConfigType", cache: Any, **kwargs) -> Any:
        """Load a config value using cached data.

        Args:
            config (Config): The config value.
            key (str): Configuration or item key.
            type (ConfigType): Target type used to cast the value.
            cache (Any): The cache value.
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Any: The operation result.
        """
        pass

    def save(self, config: "Config", key: str, value: Any) -> None:
        """Save a config value into the cache when enabled.

        Args:
            config (Config): The config value.
            key (str): Configuration or item key.
            value (Any): Value to store or process.
        """
        if self._cached:
            config.cache.save(**{key: value})


class ConfigDict(dict):

    """Dictionary wrapper that applies typed config property access."""
    def update_from_pyfile(self, filename: PathType, silent: bool = False) -> bool:
        """Load uppercase config keys from a Python file.

        Args:
            filename (PathType): File path to load.
            silent (bool): Whether missing files should be ignored.

        Returns:
            bool: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        d = ModuleType("config")
        d.__file__ = filename
        d.prompt = Config.Prompt
        d.lazy = Config.Lazy
        d.alias = Config.Alias
        d.error = Config.Error
        d.confirm = Config.Confirm
        d.prop = Config.Property
        try:
            data = utils.read_file(filename, text=False)
            exec(compile(data, filename, "exec"), d.__dict__)
        except OSError as e:
            if silent and e.errno in (errno.ENOENT, errno.EISDIR, errno.ENOTDIR):
                return False
            e.strerror = f"Unable to load configuration file ({e.strerror})"
            raise
        self.update_from_object(d)
        return True

    def update_from_file(self, filename: PathType, load: Callable[[IO[Any]], Mapping], silent: bool = False) -> bool:
        """Load config keys from a file with a custom loader.

        Args:
            filename (PathType): File path to load.
            load (Callable[[IO[Any]], Mapping]): The load value.
            silent (bool): Whether missing files should be ignored.

        Returns:
            bool: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        try:
            with open(filename, "rb") as f:
                obj = load(f)
        except OSError as e:
            if silent and e.errno in (errno.ENOENT, errno.EISDIR):
                return False

            e.strerror = f"Unable to load configuration file ({e.strerror})"
            raise

        return self.update_from_mapping(obj)

    def update_from_object(self, obj: Union[object, str]) -> None:
        """Load uppercase config keys from an object.

        Args:
            obj (Union[object, str]): Object to inspect or convert.
        """
        for key in dir(obj):
            if key[0].isupper():
                self[key] = getattr(obj, key)

    def update_from_mapping(self, mapping: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> bool:
        """Load uppercase config keys from a mapping.

        Args:
            mapping (Optional[Mapping[str, Any]]): Mapping containing values to load.
            kwargs (Any): Keyword arguments passed to the operation.

        Returns:
            bool: The operation result.
        """
        mappings: Dict[str, Any] = {}
        if mapping is not None:
            mappings.update(mapping)
        mappings.update(kwargs)
        for key, value in mappings.items():
            if key[0].isupper():
                self[key] = value
        return True


class ConfigParser(configparser.ConfigParser):

    """Parser for loading config data from a Python file."""
    def optionxform(self, optionstr: str):
        """Keep config option names unchanged.

        Args:
            optionstr (str): The optionstr value.

        Returns:
            Any: The operation result.
        """
        return optionstr


class ConfigCacheParser:

    """Parser for loading cached config data."""
    def __init__(self, path: PathType, namespace: str):
        self._parser = ConfigParser(default_section="ENV")  # Keep ENV as the legacy default section.
        self._path = path
        self._cache = FileCache(f"{self._path}.cache")
        self._section = f"{namespace}.CACHE".upper()
        self.load()

    def load(self):
        """Load cache data from disk."""
        with self._cache.backup():
            if self._path and os.path.exists(self._path):
                self._parser.read(self._path)
            if not self._parser.has_section(self._section):
                self._parser.add_section(self._section)

    def dump(self):
        """Write cache data to disk."""
        with self._cache.backup() as backup:
            if self._path and os.path.exists(self._path):
                backup.backup(self._path)
            with open(self._path, "wt") as fd:
                self._parser.write(fd)

    def get(self, key: str, default: Any) -> Any:
        """Return a cached option value.

        Args:
            key (str): Configuration or item key.
            default (Any): Value returned when no explicit value is available.

        Returns:
            Any: The operation result.
        """
        if self._parser.has_option(self._section, key):
            return self._parser.get(self._section, key)
        return default

    def set(self, key: str, value: str) -> None:
        """Set a cached option value.

        Args:
            key (str): Configuration or item key.
            value (str): Value to store or process.
        """
        self._parser.set(self._section, key, value)

    def remove(self, key: str) -> bool:
        """Remove a cached option value.

        Args:
            key (str): Configuration or item key.

        Returns:
            bool: The operation result.
        """
        return self._parser.remove_option(self._section, key)

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        """Yield cached option items.

        Returns:
            Generator[Tuple[str, Any], None, None]: The operation result.
        """
        for key, value in self._parser.items(self._section):
            yield key, value


class ConfigCache(dict):
    """Persistent key-value cache for configuration data."""
    __lock__ = threading.RLock()

    def __init__(self, environ: "BaseEnviron", namespace: str = __missing__):
        super().__init__()
        self._environ = environ
        self._namespace = namespace if namespace is not __missing__ else "MAIN"
        self._path = self._environ.get_data_path(".config", f"{self._environ.name}.cfg", create_parent=True)

        orig_path = self._environ.get_data_path(f"{self._environ.name}.cfg")
        if os.path.isfile(orig_path) and not os.path.isfile(self._path):
            self._environ.logger.warning(f"Found old config file, move `{orig_path}` to `{self._path}`")
            shutil.move(orig_path, self._path)

        self.load()

    @property
    def path(self) -> Path:
        """Return the cache file path.

        Returns:
            Path: The property value.
        """
        return self._path

    @property
    def namespace(self) -> str:
        """Return the cache namespace.

        Returns:
            str: The property value.
        """
        return self._namespace

    def load(self) -> "ConfigCache":
        """Load.

        Returns:
            ConfigCache: The operation result.
        """
        parser = ConfigCacheParser(self._path, self._namespace)
        with self.__lock__:
            self.clear()
            self.update(parser.items())
        return self

    def save(self, **kwargs: Any) -> "ConfigCache":
        """Save or download data to a target path.

        Args:
            kwargs (Any): Keyword arguments passed to the operation.

        Returns:
            ConfigCache: The operation result.
        """
        parser = ConfigCacheParser(self._path, self._namespace)
        with self.__lock__:
            for key, value in kwargs.items():
                self[key] = value
                parser.set(key, cast_str(value))
        parser.dump()
        return self

    def remove(self, *keys: str) -> "ConfigCache":
        """Remove.

        Args:
            keys (str): Keys to inspect or update.

        Returns:
            ConfigCache: The operation result.
        """
        parser = ConfigCacheParser(self._path, self._namespace)
        with self.__lock__:
            for key in keys:
                self.pop(key, None)
                parser.remove(key)
        parser.dump()
        return self


class Config:

    """Configuration container backed by defaults, files, and cache."""
    def __init__(
            self,
            environ: "BaseEnviron",
            data: ConfigDict,
            namespace: str = __missing__,
            env_prefix: str = __missing__,
    ):
        """Init.

        Args:
            environ (BaseEnviron): The environ value.
            data (ConfigDict): The data value.
            namespace (str): Argparse namespace to update.
            env_prefix (str): The env_prefix value.
        """
        self._environ = environ
        self._env_prefix = env_prefix.upper() if env_prefix is not __missing__ else ""
        self._data = data
        self._cache = ConfigCache(environ, namespace if namespace is not __missing__ else "MAIN")
        self._map = ChainMap(
            {
                key[len(self._env_prefix):]: value
                for key, value in os.environ.items()
                if key.startswith(self._env_prefix)
            },
            self._cache,
            self._data,
            self._environ.global_config,
        )

    @property
    def cache(self):
        """Return the configuration cache.

        Returns:
            Any: The property value.
        """
        return self._cache

    def reload(self) -> None:
        """Reload."""
        self._map.maps[0].clear()
        self._map.maps[0].update({
            key[len(self._env_prefix):]: value
            for key, value in os.environ.items()
            if key.startswith(self._env_prefix)
        })
        self._cache.clear()

    def cast(self, obj: Any, type: "ConfigType", default: Any = __missing__) -> "T":
        """Cast a value to the requested type.

        Args:
            obj (Any): Object to inspect or convert.
            type (ConfigType): Target type used to cast the value.
            default (Any): Value returned when no explicit value is available.

        Returns:
            T: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        if type not in (None, __missing__):
            cast = CONFIG_TYPES.get(type, type)
            try:
                return cast(obj)
            except Exception:
                if default is not __missing__:
                    return default
                raise
        return obj

    def get(self, key: str, type: "ConfigType" = None, default: Any = __missing__) -> "T":
        """Get.

        Args:
            key (str): Configuration or item key.
            type (ConfigType): Target type used to cast the value.
            default (Any): Value returned when no explicit value is available.

        Returns:
            T: The operation result.

        Raises:
            Exception: Propagates errors raised while completing the operation.
        """
        if type in (None, __missing__):
            value = self._data.get(key, __missing__)
            if isinstance(value, ConfigProperty):
                type = value.type

        try:
            value = self._map.get(key, __missing__)
            if value is not __missing__:
                if isinstance(value, ConfigProperty):
                    with self._cache.__lock__:
                        result = self._cache[key] = value.get(self, key, type=type, default=__missing__)
                        return result
                return self.cast(value, type=type)
            raise ConfigError(f"Not found environment variable \"{self._env_prefix}{key}\" or config \"{key}\"")
        except ConfigError:
            if default is __missing__:
                raise
        except Exception as e:
            if self._environ.debug:
                self._environ.logger.debug(f"Failed to get config \"{key}\"", exc_info=True)
            if default is __missing__:
                raise ConfigError(f"Failed to get config \"{key}\"") from e

        if isinstance(default, ConfigProperty):
            try:
                with self._cache.__lock__:
                    result = self._cache[key] = default.get(self, key, type=type, default=__missing__)
                    return result
            except ConfigError:
                raise
            except Exception as e:
                if self._environ.debug:
                    self._environ.logger.debug(f"Failed to get default config \"{key}\"", exc_info=True)
                raise ConfigError(f"Failed to get default config \"{key}\"") from e

        return default

    def keys(self) -> Generator[str, None, None]:
        """Keys.

        Returns:
            Generator[str, None, None]: The operation result.
        """
        for key in sorted(self._map.keys()):
            yield key

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        """Items.

        Returns:
            Generator[Tuple[str, Any], None, None]: The operation result.
        """
        for key in self.keys():
            yield key, self.get(key)

    def set(self, key: str, value: Any) -> "Config":
        """Set.

        Args:
            key (str): Configuration or item key.
            value (Any): Value to store or process.

        Returns:
            Config: The operation result.
        """
        self._data[key] = value
        return self

    def set_default(self, key: str, value: Any) -> Any:
        """Set the default.

        Args:
            key (str): Configuration or item key.
            value (Any): Value to store or process.

        Returns:
            Any: The operation result.
        """
        return self._data.setdefault(key, value)

    def update(self, **kwargs) -> "Config":
        """Update.

        Args:
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Config: The operation result.
        """
        self._data.update(**kwargs)
        return self

    def update_defaults(self, **kwargs) -> "Config":
        """Update defaults.

        Args:
            kwargs: Keyword arguments passed to the operation.

        Returns:
            Config: The operation result.
        """
        for key, value in kwargs.items():
            self._data.setdefault(key, value)
        return self

    def update_from_file(self, path: str, load: Callable[[IO[Any]], Mapping] = None) -> bool:
        """Update from file.

        Args:
            path (str): Filesystem path to process.
            load (Callable[[IO[Any]], Mapping]): The load value.

        Returns:
            bool: The operation result.
        """
        if load is not None:
            return self._data.update_from_file(path, load=load)
        if path.endswith(".py"):
            return self._data.update_from_pyfile(path)
        elif path.endswith(".json"):
            return self._data.update_from_file(path, load=json.load)
        self._environ.logger.debug(f"Unsupported config file: {path}")
        return False

    def update_from_dir(self, path: str, recursion: bool = False) -> bool:
        """Update from dir.

        Args:
            path (str): Filesystem path to process.
            recursion (bool): The recursion value.

        Returns:
            bool: The operation result.
        """
        # Path does not exist.
        if not os.path.exists(path):
            return False
        # Non-directory paths are loaded as a single file.
        if not os.path.isdir(path):
            return self.update_from_file(path)
        # Without recursion, only load files directly inside the directory.
        if not recursion:
            for name in os.listdir(path):
                config_path = os.path.join(path, name)
                if not os.path.isdir(config_path):
                    self.update_from_file(config_path)
            return True
        # Recursively load all files under the directory.
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                self.update_from_file(os.path.join(root, name))
        return True

    def update_cache(self, **kwargs: Any) -> "Config":
        """Update cache.

        Args:
            kwargs (Any): Keyword arguments passed to the operation.

        Returns:
            Config: The operation result.
        """
        self._cache.save(**kwargs)
        return self

    def remove_cache(self, *keys: str) -> "Config":
        """Remove cache.

        Args:
            keys (str): Keys to inspect or update.

        Returns:
            Config: The operation result.
        """
        self._cache.remove(*keys)
        return self

    def __contains__(self, key) -> bool:
        return key in self._map

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any):
        self.set(key, value)

    class Property(CacheConfigProperty):

        def __init__(
                self,
                key: str = __missing__,
                type: "ConfigType" = str,
                default: Any = __missing__,
                cached: bool = __missing__,
        ):
            super().__init__(type=type, default=default, cached=cached)
            self.key = key

        def get(self, config: "Config", key: str, type: "ConfigType", default: Any, **kwargs) -> Any:
            return super().get(config, self.key or key, type, default, **kwargs)

        def load(self, config: "Config", key: str, *, type: "ConfigType", cache: Any, **kwargs) -> Any:
            if isinstance(self.default, ConfigProperty):
                return self.default.get(
                    config,
                    self.key or key,
                    type=type or self.type,
                    default=cache
                )
            if self.key is __missing__:  # Avoid recursion.
                return cache if cache is not __missing__ else self.default
            return config.get(
                self.key or key,
                type=type or self.type,
                default=cache if cache is not __missing__ else self.default
            )

        def save(self, config: "Config", key: str, value: Any) -> None:
            if self._cached is __missing__:
                if isinstance(self.default, CacheConfigProperty):
                    return self.default.save(config, self.key or key, value)
            return super().save(config, self.key or key, value)

    class Prompt(CacheConfigProperty):

        type: "Union[Type[Union[str, int, float]], ConfigLiteralType]"

        def __init__(
                self,
                prompt: str = None,
                password: bool = False,
                choices: "Union[List[str], Dict[str, str]]" = None,
                type: "Union[Type[Union[str, int, float]], ConfigLiteralType]" = str,
                default: Any = __missing__,
                cached: bool = __missing__,
                always_ask: bool = False,
                allow_empty: bool = False,
        ):
            super().__init__(type=type, default=default, cached=cached)

            self.prompt = prompt
            self.password = password
            self.choices = choices
            self.always_ask = always_ask
            self.allow_empty = allow_empty

        def load(self, config: "Config", key: str, type: "ConfigType", cache: Any,
                 choices: "Union[List[str], Dict[str, str]]" = None,
                 **kwargs):

            if cache is not __missing__ and not self.always_ask:
                if key in config.cache:
                    return cache

            default = cache
            if default is __missing__:
                default = self.default
                if isinstance(default, ConfigProperty):
                    default = default.get(config, key, type=type or self.type, default=cache)

            if default is not __missing__:
                default = config.cast(default, self.type)

            choices = choices or self.choices
            if choices:
                return choose(
                    self.prompt or f"Please choose {key}",
                    choices=choices,
                    default=default,
                    show_default=True,
                    show_choices=True
                )

            return prompt(
                self.prompt or f"Please enter {key}",
                type=self.type if not isinstance(self.type, str) else str,
                password=self.password,
                default=default,
                allow_empty=self.allow_empty,
                show_default=True,
                show_choices=True
            )

    class Confirm(CacheConfigProperty):

        def __init__(
                self,
                prompt: str = None,
                default: Any = __missing__,
                cached: bool = __missing__,
                always_ask: bool = False,
        ):
            super().__init__(type=bool, default=default, cached=cached)

            self.prompt = prompt
            self.always_ask = always_ask

        def load(self, config: "Config", key: str, type: "ConfigType", cache: Any, **kwargs):

            if cache is not __missing__ and not self.always_ask:
                if key in config.cache:
                    return cache

            default = cache
            if default is __missing__:
                default = self.default
                if isinstance(default, ConfigProperty):
                    default = default.get(config, key, type=type or self.type, default=cache)

            if default is not __missing__:
                default = config.cast(default, bool)

            return confirm(
                self.prompt or f"Please confirm {key}",
                default=default,
                show_default=True,
            )

    class Alias(CacheConfigProperty):

        def __init__(
                self,
                *keys: str,
                type: "ConfigType" = str,
                default: Any = __missing__,
                cached: bool = __missing__
        ):
            super().__init__(type=type, default=default, cached=cached)
            self.keys = keys

        def load(self, config: "Config", key: str, type: "ConfigType", cache: Any, **kwargs):
            if cache is not __missing__:
                return cache

            if self.default is __missing__:
                last_error = None
                for key in self.keys:
                    try:
                        return config.get(key, type=type or self.type)
                    except Exception as e:
                        last_error = e

                raise last_error or ConfigError(f"Cannot find config \"{key}\"")

            else:
                for key in self.keys:
                    result = config.get(key, type=type or self.type, default=SUPPRESS)
                    if result is not SUPPRESS:
                        return result

                return self.default

    class Lazy(LazyConfigProperty):

        def __init__(self, func: "Callable[[Config], T]"):
            super().__init__()
            self.func = func

        def load(self, config: "Config", key: str, **kwargs) -> Any:
            return self.func(config)

    class Error(LazyConfigProperty):

        def __init__(self, message: str = None):
            super().__init__()
            self.message = message

        def load(self, config: "Config", key: str, **kwargs) -> Any:
            message = self.message or \
                      f"Cannot find config \"{key}\". {os.linesep}" \
                      f"You can use any of the following methods to fix it: {os.linesep}" \
                      f"1. set \"{config._env_prefix}{key}\" as an environment variable, {os.linesep}" \
                      f"2. call config.cache.save({key}=xxx) method to save the value to file. {os.linesep}"
            raise ConfigError(message)


class ConfigWrapper(Config):

    """Proxy wrapper that exposes a scoped configuration view."""
    def __init__(
            self,
            config: "Config",
            namespace: str = __missing__,
            env_prefix: str = __missing__,
    ):
        super().__init__(
            config._environ,
            config._data,
            namespace=namespace if namespace is not __missing__ else config.cache.namespace,
            env_prefix=env_prefix if env_prefix is not __missing__ else config._env_prefix
        )
