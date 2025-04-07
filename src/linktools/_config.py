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
import threading
from collections import ChainMap
from pathlib import Path
from types import ModuleType
from typing import \
    TYPE_CHECKING, TypeVar, Type, Optional, Generator, \
    Any, Tuple, IO, Mapping, Union, List, Dict, Callable

from . import utils
from .metadata import __missing__
from .rich import choose, prompt, confirm
from .types import PathType, get_args, ConfigError

if TYPE_CHECKING:
    from typing import Literal, Self
    from ._environ import BaseEnviron

    T = TypeVar("T")
    EnvironType = TypeVar("EnvironType", bound=BaseEnviron)
    LiteralType = Literal["path", "json"]
    ConfigType = Union[Type[T], LiteralType]

SUPPRESS = object()


def is_type(obj: Any) -> bool:
    return isinstance(obj, type)


def cast_bool(obj: Any) -> bool:
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
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (Tuple, List, Dict)):
        return json.dumps(obj)
    if obj is None:
        return ""
    return str(obj)


def cast_path(obj: Any) -> str:
    if isinstance(obj, get_args(PathType)):
        return os.path.abspath(
            os.path.expanduser(
                str(obj)  # support Proxy object
            )
        )
    raise TypeError(f"{type(obj)} cannot be converted to path")


def cast_json(obj: Any) -> Union[List, Dict]:
    if isinstance(obj, str):
        return json.loads(obj)
    if isinstance(obj, (Tuple, List, Dict)):
        return obj
    raise TypeError(f"{type(obj)} cannot be converted to json")


CONFIG_TYPES: "Dict[ConfigType, Callable[[Any], T]]" = dict({
    bool: cast_bool,
    str: cast_str,
    "path": cast_path,
    "json": cast_json,
})


class ConfigProperty(metaclass=abc.ABCMeta):

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
        return self._type

    @property
    def default(self) -> Any:
        return self._default

    @abc.abstractmethod
    def get(self, config: "Config", key: str, *, type: "ConfigType", default: Any, **kwargs) -> Any:
        pass

    def set_default(self, value: Any, ignore_errors: bool = False) -> "Self":
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

    def get(self, config: "Config", key: str, *, type: "ConfigType", default: Any, **kwargs) -> Any:
        result = self.load(config, key, type=type, default=default, **kwargs)
        if isinstance(result, ConfigProperty):
            result = result.get(config, key, type=type, default=default, **kwargs)
        return result

    @abc.abstractmethod
    def load(self, config: "Config", key: str, *, type: "ConfigType", default: Any, **kwargs) -> Any:
        pass


class CacheConfigProperty(ConfigProperty, metaclass=abc.ABCMeta):

    def __init__(self, *, type: "ConfigType" = None, default: Any, cached: bool = __missing__):
        super().__init__(type=type, default=default)
        self._data = __missing__
        self._cached = cached

    def get(self, config: "Config", key: str, type: "ConfigType", default: Any, **kwargs) -> Any:
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
        pass

    def save(self, config: "Config", key: str, value: Any) -> None:
        if self._cached:
            config.cache.save(**{key: value})


class ConfigDict(dict):

    def update_from_pyfile(self, filename: PathType, silent: bool = False) -> bool:
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
        for key in dir(obj):
            if key[0].isupper():
                self[key] = getattr(obj, key)

    def update_from_mapping(self, mapping: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> bool:
        mappings: Dict[str, Any] = {}
        if mapping is not None:
            mappings.update(mapping)
        mappings.update(kwargs)
        for key, value in mappings.items():
            if key[0].isupper():
                self[key] = value
        return True


class ConfigParser(configparser.ConfigParser):

    def optionxform(self, optionstr: str):
        return optionstr


class ConfigCacheParser:

    def __init__(self, path: PathType, namespace: str):
        self._parser = ConfigParser(default_section="ENV")  # 兼容老版本，默认ENV作为默认节
        self._path = path
        self._section = f"{namespace}.CACHE".upper()
        self.load()

    def load(self):
        if self._path and os.path.exists(self._path):
            self._parser.read(self._path)
        if not self._parser.has_section(self._section):
            self._parser.add_section(self._section)

    def dump(self):
        with open(self._path, "wt") as fd:
            self._parser.write(fd)

    def get(self, key: str, default: Any) -> Any:
        if self._parser.has_option(self._section, key):
            return self._parser.get(self._section, key)
        return default

    def set(self, key: str, value: str) -> None:
        self._parser.set(self._section, key, value)

    def remove(self, key: str) -> bool:
        return self._parser.remove_option(self._section, key)

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        for key, value in self._parser.items(self._section):
            yield key, value


class ConfigCache(dict):
    __lock__ = threading.RLock()

    def __init__(self, environ: "BaseEnviron", namespace: str = __missing__):
        super().__init__()
        self._environ = environ
        self._namespace = namespace if namespace is not __missing__ else "MAIN"
        self._path = self._environ.get_data_path(f"{self._environ.name}.cfg", create_parent=True)
        self.load()

    @property
    def path(self) -> Path:
        """
        缓存文件路径
        """
        return self._path

    @property
    def namespace(self) -> str:
        """
        缓存命名空间
        """
        return self._namespace

    def load(self) -> "ConfigCache":
        """
        从缓存中加载配置
        """
        parser = ConfigCacheParser(self._path, self._namespace)
        with self.__lock__:
            self.clear()
            self.update(parser.items())
        return self

    def save(self, **kwargs: Any) -> "ConfigCache":
        """
        保存配置到缓存
        :param kwargs: 需要保存的配置
        """
        parser = ConfigCacheParser(self._path, self._namespace)
        with self.__lock__:
            for key, value in kwargs.items():
                self[key] = value
                parser.set(key, cast_str(value))
        parser.dump()
        return self

    def remove(self, *keys: str) -> "ConfigCache":
        """
        删除缓存
        :param keys: 需要删除的缓存键
        """
        parser = ConfigCacheParser(self._path, self._namespace)
        with self.__lock__:
            for key in keys:
                self.pop(key, None)
                parser.remove(key)
        parser.dump()
        return self


class Config:

    def __init__(
            self,
            environ: "BaseEnviron",
            data: ConfigDict,
            namespace: str = __missing__,
            env_prefix: str = __missing__,
    ):
        """
        初始化配置对象
        :param environ: 环境对象
        :param data: 配置相关数据
        :param namespace: 缓存对应的命名空间
        :param env_prefix: 环境变量前缀
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
        )

    @property
    def cache(self):
        """
        缓存对象
        """
        return self._cache

    def reload(self) -> None:
        """
        重新加载配置，包括：清空缓存、刷新环境变量
        """
        self._map.maps[0].clear()
        self._map.maps[0].update({
            key[len(self._env_prefix):]: value
            for key, value in os.environ.items()
            if key.startswith(self._env_prefix)
        })
        self._cache.clear()

    def cast(self, obj: Any, type: "ConfigType", default: Any = __missing__) -> "T":
        """
        类型转换
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
        """
        获取指定配置，优先会从环境变量中获取
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
                raise ConfigError(f"Failed to get default config \"{key}\"") from e

        return default

    def keys(self) -> Generator[str, None, None]:
        """
        遍历配置名，默认不遍历内置配置
        """
        for key in sorted(self._map.keys()):
            yield key

    def items(self) -> Generator[Tuple[str, Any], None, None]:
        """
        遍历配置项，默认不遍历内置配置
        """
        for key in self.keys():
            yield key, self.get(key)

    def set(self, key: str, value: Any) -> "Config":
        """
        更新配置
        """
        self._data[key] = value
        return self

    def set_default(self, key: str, value: Any) -> Any:
        """
        设置默认配置
        """
        return self._data.setdefault(key, value)

    def update(self, **kwargs) -> "Config":
        """
        更新配置
        """
        self._data.update(**kwargs)
        return self

    def update_defaults(self, **kwargs) -> "Config":
        """
        更新默认配置
        """
        for key, value in kwargs.items():
            self._data.setdefault(key, value)
        return self

    def update_from_file(self, path: str, load: Callable[[IO[Any]], Mapping] = None) -> bool:
        """
        加载配置文件，按照扩展名来匹配相应的加载规则
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
        """
        加载配置文件目录，按照扩展名来匹配相应的加载规则
        """
        # 路径不存在
        if not os.path.exists(path):
            return False
        # 如果不是目录
        if not os.path.isdir(path):
            return self.update_from_file(path)
        # 如果不需要递归，那只要取一级目录就好了
        if not recursion:
            for name in os.listdir(path):
                config_path = os.path.join(path, name)
                if not os.path.isdir(config_path):
                    self.update_from_file(config_path)
            return True
        # 剩下的就是需要递归读取所有文件的情况了
        for root, dirs, files in os.walk(path, topdown=False):
            for name in files:
                self.update_from_file(os.path.join(root, name))
        return True

    def update_cache(self, **kwargs: Any) -> "Config":
        """
        更新缓存
        """
        self._cache.save(**kwargs)
        return self

    def remove_cache(self, *keys: str) -> "Config":
        """
        删除缓存
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

        def __init__(
                self,
                prompt: str = None,
                password: bool = False,
                choices: "Union[List[str], Dict[str, str]]" = None,
                type: "Union[Type[Union[str, int, float]], LiteralType]" = str,
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

            default = cache
            if default is not __missing__ and not self.always_ask:
                return default

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

            default = cache
            if default is not __missing__ and not self.always_ask:
                return default

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
                if last_error is not None:
                    raise last_error

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
