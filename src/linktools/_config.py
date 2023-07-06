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
   /  oooooooooooooooo  .o.  oooo /,   \,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,`\--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""

import abc
import errno
import json
import os
import pickle
import threading
from types import ModuleType
from typing import Type, Optional, Any, Dict, Generator, Tuple, Callable, IO, Mapping, Union, List, TypeVar

from rich.prompt import Prompt, IntPrompt, FloatPrompt, Confirm, PromptType, InvalidResponse

from . import utils
from ._environ import BaseEnviron

T = TypeVar("T")
MISSING = ...


def _cast_bool(obj: Any) -> bool:
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        data = obj.lower()
        if data in ("true", "yes", "y", "1"):
            return True
        elif data in ("false", "no", "n", "0"):
            return False
        raise TypeError(f"str '{obj}' cannot be converted to type bool")
    return bool(obj)


def _cast_str(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    if obj is None:
        return ""
    return str(obj)


CAST_DICT: Dict[Type[T], Callable[..., T]] = {
    bool: _cast_bool,
    str: _cast_str,
}


class ConfigError(Exception):
    pass


class ConfigProperty(abc.ABC):
    __lock__ = threading.RLock()

    def __init__(self, type: Type = None, cached: bool = False):
        self._data: Union[str, object] = MISSING
        self._type = type
        self._cached = cached

    def load(self, env: BaseEnviron, key: str, type: Type = None) -> Any:
        if self._data is not MISSING:
            return self._data
        with self.__lock__:
            if self._data is not MISSING:
                return self._data
            type = type or self._type
            if self._cached:
                cache = MISSING
                path = env.get_data_path("configs", f"cached_{env.name}", key, create_parent=True)
                if os.path.exists(path):
                    cache = utils.read_file(path, binary=False)
                result = self._load(env, key, cache)
                if isinstance(result, ConfigProperty):
                    result = result.load(env, key, type=type)
                elif type and not isinstance(result, type):
                    result = env.config.cast(result, type)
                utils.write_file(path, str(result))
                self._data = result
            else:
                result = self._load(env, key, MISSING)
                if isinstance(result, ConfigProperty):
                    result = result.load(env, key, type=type)
                elif type and not isinstance(result, type):
                    result = env.config.cast(result, type)
                self._data = result
            return self._data

    @abc.abstractmethod
    def _load(self, env: BaseEnviron, key: Any, cache: Any):
        pass


class ConfigDict(dict):

    def update_from_pyfile(self, filename: str, silent: bool = False) -> bool:
        d = ModuleType("config")
        d.__file__ = filename
        d.prompt = Config.Prompt
        d.lazy = Config.Lazy
        d.alias = Config.Alias
        d.error = Config.Error
        d.confirm = Config.Confirm
        try:
            with open(filename, "rb") as config_file:
                exec(compile(config_file.read(), filename, "exec"), d.__dict__)
        except OSError as e:
            if silent and e.errno in (errno.ENOENT, errno.EISDIR, errno.ENOTDIR):
                return False
            e.strerror = f"Unable to load configuration file ({e.strerror})"
            raise
        self.update_from_object(d)
        return True

    def update_from_file(self, filename: str, load: Callable[[IO[Any]], Mapping], silent: bool = False) -> bool:
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


class Config:

    def __init__(self, env: BaseEnviron, internal: ConfigDict):
        self._environ = env
        self._internal = internal
        self._config = pickle.loads(pickle.dumps(self._internal))
        self._envvar_prefix = f"{self._environ.name.upper()}_"

    @property
    def envvar_prefix(self):
        """
        环境变量前缀
        """
        return self._envvar_prefix

    @envvar_prefix.setter
    def envvar_prefix(self, value: str):
        """
        环境变量前缀
        """
        self._envvar_prefix = value

    def cast(self, obj: str, type: Type[T] = None):
        cast = CAST_DICT.get(type, type)
        return cast(obj) if cast is not None else obj

    def get_namespace(self, namespace: str, lowercase: bool = True, trim_namespace: bool = True) -> Dict[str, Any]:
        """
        根据命名空间获取配置列表
        """
        rv = {}
        for k in self._config:
            if not k.startswith(namespace):
                continue
            if trim_namespace:
                key = k[len(namespace):]
            else:
                key = k
            if lowercase:
                key = key.lower()
            rv[key] = self.get(k)
        return rv

    def get(self, key: str, type: Type[T] = None, default: Union[T, ConfigProperty] = MISSING) -> T:
        """
        获取指定配置，优先会从环境变量中获取
        """

        last_error = MISSING
        try:

            env_key = f"{self.envvar_prefix}{key}"
            if env_key in os.environ:
                value = os.environ.get(env_key)
                return self.cast(value, type=type)

            if key in self._config:
                value = self._config.get(key)
                if isinstance(value, ConfigProperty):
                    return value.load(self._environ, key, type=type)
                return self.cast(value, type=type)

        except Exception as e:
            last_error = e

        if default is MISSING:
            if last_error is not MISSING:
                raise last_error
            raise ConfigError(f"Not found environment variable \"{self.envvar_prefix}{key}\" or config \"{key}\"")

        if isinstance(default, ConfigProperty):
            return default.load(self._environ, key, type=type)

        return default

    def get_str(self, key: str, default: Union[str, ConfigProperty] = MISSING) -> str:
        return self.get(key, type=str, default=default)

    def get_bool(self, key: str, default: Union[bool, ConfigProperty] = MISSING) -> bool:
        return self.get(key, type=bool, default=default)

    def get_int(self, key: str, default: Union[int, ConfigProperty] = MISSING) -> int:
        return self.get(key, type=int, default=default)

    def get_float(self, key: str, default: Union[float, ConfigProperty] = MISSING) -> float:
        return self.get(key, type=float, default=default)

    def keys(self, all: bool = False) -> Generator[str, None, None]:
        """
        遍历配置名，默认不遍历内置配置
        """
        for key in self._config.keys():
            if all or key not in self._internal:
                yield key

    def items(self, all: bool = False) -> Generator[Tuple[str, Any], None, None]:
        """
        遍历配置项，默认不遍历内置配置
        """
        for key in self.keys(all=all):
            yield key, self.get(key)

    def set(self, key: str, value: Any) -> None:
        """
        更新配置
        """
        self._config[key] = value

    def set_default(self, key: str, value: Any) -> Any:
        """
        设置默认配置
        """
        return self._config.setdefault(key, value)

    def update(self, **kwargs) -> None:
        """
        更新配置
        """
        self._config.update(**kwargs)

    def update_defaults(self, **kwargs) -> None:
        """
        更新默认配置
        """
        for key, value in kwargs.items():
            self._config.setdefault(key, value)

    def update_from_file(self, path: str, load: Callable[[IO[Any]], Mapping] = None) -> bool:
        """
        加载配置文件，按照扩展名来匹配相应的加载规则
        """
        if load is not None:
            return self._config.update_from_file(path, load=load)
        if path.endswith(".py"):
            return self._config.update_from_pyfile(path)
        elif path.endswith(".json"):
            return self._config.update_from_file(path, load=json.load)
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

    def update_from_envvar(self, prefix: str = None) -> bool:
        """
        加载所有以"{prefix}"为前缀的环境变量到配置中
        """
        if prefix is None:
            prefix = self.envvar_prefix
        for key, value in os.environ.items():
            if key.startswith(prefix):
                self._config[key[len(prefix):]] = value
        return True

    class Prompt(ConfigProperty):

        _prompt_types: Dict[Type, Type[Prompt]] = {
            str: Prompt,
            int: IntPrompt,
            float: FloatPrompt,
        }

        @classmethod
        def _create_prompt_class(cls, type: Type, allow_empty: bool):
            base_prompt_types = cls._prompt_types.get(type)
            if base_prompt_types is None:
                support_types = [str(key) for key in cls._prompt_types.keys()]
                raise ConfigError(f"prompt only supports {support_types} type")

            class ConfigPrompt(base_prompt_types):

                def process_response(self, value: str) -> PromptType:
                    value = value.strip()
                    if not allow_empty and utils.is_empty(value):
                        raise InvalidResponse(self.validate_error_message)
                    return super().process_response(value)

            return ConfigPrompt

        def __init__(
                self,
                prompt: str = None,
                password: bool = False,
                choices: Optional[List[str]] = None,
                type: Type = str,
                default: Any = MISSING,
                cached: bool = False,
                always_ask: bool = False,
                allow_empty: bool = False,
        ):
            super().__init__(type=type, cached=cached)

            self.prompt_class = self._create_prompt_class(type, allow_empty)
            self.prompt = prompt
            self.password = password
            self.choices = choices
            self.default = default
            self.always_ask = always_ask

        def _load(self, env: BaseEnviron, key: Any, cache: Any):

            default = cache
            if default is not MISSING and not self.always_ask:
                if not env.get_config("RELOAD_CONFIG", type=bool):
                    return default

            if default is MISSING:
                default = self.default
                if isinstance(default, ConfigProperty):
                    default = default.load(env, key)

            return self.prompt_class.ask(
                self.prompt or f"Please input {key}",
                password=self.password,
                choices=self.choices,
                default=default,
                show_default=True,
                show_choices=True
            )

    class Confirm(ConfigProperty):

        def __init__(
                self,
                prompt: str = None,
                default: Any = MISSING,
                cached: bool = False,
                always_ask: bool = False,
        ):
            super().__init__(type=bool, cached=cached)

            self.prompt_class = Confirm
            self.prompt = prompt
            self.default = default
            self.always_ask = always_ask

        def _load(self, env: BaseEnviron, key: Any, cache: Any):

            default = cache
            if default is not MISSING and not self.always_ask:
                if not env.get_config("RELOAD_CONFIG", type=bool):
                    return default

            if default is MISSING:
                default = self.default
                if isinstance(default, ConfigProperty):
                    default = default.load(env, key)

            return self.prompt_class.ask(
                self.prompt or f"Please confirm {key}",
                default=default,
                show_default=True,
            )

    class Alias(ConfigProperty):

        _not_found = object()

        def __init__(self, key: str, type: Type = str, default: Any = MISSING, cached: bool = False):
            super().__init__(type=type, cached=cached)
            self.key = key
            self.default = default

        def _load(self, env: BaseEnviron, key: Any, cache: Any):
            if cache is not MISSING:
                return cache

            if self.default is MISSING:
                return env.get_config(self.key)

            result = env.get_config(self.key, default=self._not_found)
            if result is self._not_found:
                result = self.default

            return result

    class Lazy(ConfigProperty):

        def __init__(self, func: Callable[[BaseEnviron], Any]):
            super().__init__()
            self.func = func

        def _load(self, env: BaseEnviron, key: Any, cache: Any):
            return self.func(env)

    class Error(ConfigProperty):

        def __init__(self, message: str = None):
            super().__init__()
            self.message = message

        def _load(self, env: BaseEnviron, key: Any, cache: Any):
            raise ConfigError(
                self.message or
                f"Please set \"{env.config.envvar_prefix}{key}\" as an environment variable, "
                f"or set \"{key}\" in config file"
            )