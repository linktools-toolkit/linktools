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
from typing import TYPE_CHECKING, TypeVar, Type, Any

from . import utils, metadata
from .decorator import cached_property, cached_classproperty

if TYPE_CHECKING:
    from ._config import ConfigDict, Config
    from ._tools import Tools, Tool
    from ._url import UrlFile
    from .types import PathType

    T = TypeVar("T")


class BaseEnviron(abc.ABC):

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """
        模块名
        """
        pass

    @property
    def version(self) -> str:
        """
        模块版本号
        """
        return NotImplemented

    @property
    def description(self) -> str:
        """
        模块描述
        """
        return NotImplemented

    @property
    def root_path(self) -> "PathType":
        """
        模块路径
        """
        raise NotImplemented

    @property
    def system(self) -> str:
        """
        系统名称
        """
        return utils.get_system()

    @property
    def machine(self) -> str:
        """
        机器类型，e.g. 'i386'
        """
        return utils.get_machine()

    @property
    def debug(self) -> bool:
        """
        debug模式
        """
        return self.get_config("DEBUG", type=bool)

    @debug.setter
    def debug(self, value: bool) -> None:
        """
        debug模式
        """
        self.set_config("DEBUG", value)

    @cached_property
    def data_path(self) -> Path:
        """
        存放文件目录
        """
        prefix = f"{metadata.__name__}".upper()
        path = os.environ.get(f"{prefix}_DATA_PATH", None)
        if path:  # 优先使用环境变量中的${DATA_PATH}
            return Path(path)
        path = os.environ.get(f"{prefix}_STORAGE_PATH", None)
        if path:  # 其次使用环境变量中的${STORAGE_PATH}/data
            return Path(path, "data")
        # 最后使用默认路径${HOME}/.linktools/data
        return Path.home().joinpath(f".{metadata.__name__}", "data")

    @cached_property
    def temp_path(self) -> Path:
        """
        存放临时文件目录
        """
        prefix = f"{metadata.__name__}".upper()
        path = os.environ.get(f"{prefix}_TEMP_PATH", None)
        if path:  # 优先使用环境变量中的${TEMP_PATH}
            return Path(path)
        path = os.environ.get(f"{prefix}_STORAGE_PATH", None)
        if path:  # 其次使用环境变量中的${STORAGE_PATH}/temp
            return Path(path, "temp")
        # 最后使用默认路径${HOME}/.linktools/temp
        return Path.home().joinpath(f".{metadata.__name__}", "temp")

    def get_path(self, *paths: str) -> Path:
        """
        获取模块目录下的子路径
        """
        if self.root_path == NotImplemented:
            raise RuntimeError("root_path not implemented")
        return utils.join_path(self.root_path, *paths)

    def get_data_path(self, *paths: str, create_parent: bool = False) -> Path:
        """
        获取数据目录下的子路径
        """
        path = utils.join_path(self.data_path, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def get_temp_path(self, *paths: str, create_parent: bool = False) -> Path:
        """
        获取临时文件目录下的子路径
        """
        path = utils.join_path(self.temp_path, *paths)
        if create_parent:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def clean_temp_files(self, *paths: str, expire_days: int = 7) -> None:
        """
        清理临时文件
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

    @cached_classproperty(lock=True)
    def _log_manager(self) -> "logging.Manager":

        empty_args = tuple()

        class Logger(logging.Logger):

            def _log(self, level, msg, args, **kwargs):
                msg = str(msg)
                msg += ''.join([str(i) for i in args])

                kwargs["extra"] = kwargs.get("extra") or {}
                self._move_args(
                    kwargs, kwargs["extra"],
                    "style", "indent", "markup", "highlighter"
                )

                return super()._log(level, msg, empty_args, **kwargs)

            @classmethod
            def _move_args(cls, from_, to_, *keys):
                for key in keys:
                    value = from_.pop(key, None)
                    if value is not None:
                        to_[key] = value

        class LogManager(utils.get_derived_type(logging.Manager)):

            def __init__(self, manager):
                super().__init__(manager)
                object.__setattr__(self, "loggerClass", Logger)

            def getLogger(self, name):
                return logging.Manager.getLogger(self, name)

        return LogManager(logging.root.manager)

    @cached_property
    def logger(self) -> "logging.Logger":
        """
        模块根logger
        """
        return self._log_manager.getLogger(self.name)

    def get_logger(self, name: str = None) -> "logging.Logger":
        """
        获取模块名作为前缀的logger
        """
        name = f"{self.name}.{name}" if name else self.name
        return self._log_manager.getLogger(name)

    def _create_config(self) -> "Config":
        from ._config import Config, ConfigDict

        return Config(
            self,
            ConfigDict(
                DEBUG=False,
            ),
            namespace="MAIN",
            env_prefix=f"{self.name.upper()}_"
        )

    @cached_property(lock=True)
    def config(self) -> "Config":
        """
        环境相关配置
        """
        return self._create_config()

    def wrap_config(self, namespace: str = metadata.__missing__, env_prefix: str = metadata.__missing__) -> "Config":
        """
        环境相关配置，与environ.config共享配置数据，但不共享缓存数据和环境变量信息
        :param namespace: 缓存对应的命名空间
        :param env_prefix: 环境变量使用前缀
        :return: 配置对象
        """
        from ._config import ConfigWrapper

        return ConfigWrapper(self.config, namespace=namespace, env_prefix=env_prefix)

    def get_config(self, key: str, type: "Type[T]" = None, default: Any = metadata.__missing__) -> "T":
        """
        获取指定配置，优先会从环境变量中获取
        :param key: 配置键
        :param type: 配置类型
        :param default: 默认值
        :return: 配置值
        """
        return self.config.get(key=key, type=type, default=default)

    def set_config(self, key: str, value: Any) -> None:
        """
        更新配置
        :param key: 配置键
        :param value: 配置值
        """
        self.config.set(key, value)

    def _create_tools(self) -> "Tools":
        from ._tools import Tools
        from ._config import ConfigDict

        config = ConfigDict()

        template_path = environ.get_path("template", "tools.yml")
        data_path = environ.get_data_path("tools", "tools.json")
        asset_path = environ.get_asset_path("tools.json")

        if os.path.exists(data_path):
            config.update_from_file(data_path, json.load)
        elif metadata.__release__ or not os.path.exists(template_path):
            config.update_from_file(asset_path, json.load)
        else:
            import yaml
            config.update_from_file(template_path, yaml.safe_load)

        tools = Tools(self, config)
        paths = os.environ["PATH"].split(os.pathsep)
        stub_path = str(tools.stub_path)
        if stub_path not in paths:
            paths.append(stub_path)
            os.environ["PATH"] = os.pathsep.join(paths)
        return tools

    @cached_property(lock=True)
    def tools(self) -> "Tools":
        """
        工具集
        """
        return self._create_tools()

    def get_tool(self, name: str, **kwargs) -> "Tool":
        """
        获取指定工具
        :param name: 工具名
        :param kwargs: 工具其他参数
        :return: 工具对象
        """
        tool = self.tools[name]
        if len(kwargs) != 0:
            tool = tool.copy(**kwargs)
        return tool

    def get_url_file(self, url: "PathType") -> "UrlFile":
        """
        获取指定url
        :param url: url地址
        :return: UrlFile对象
        """
        from ._url import HttpFile, LocalFile

        if not isinstance(url, str):
            url = str(url)

        if url.startswith("http://") or url.startswith("https://"):
            return HttpFile(self, url)
        elif url.startswith("file://"):
            return LocalFile(self, url[len("file://"):])

        return LocalFile(self, url)


class Environ(BaseEnviron):

    @property
    def name(self) -> str:
        return metadata.__name__

    @property
    def version(self) -> str:
        return metadata.__version__

    @property
    def description(self) -> str:
        return metadata.__description__

    @cached_property
    def root_path(self) -> str:
        return os.path.dirname(__file__)

    def get_asset_path(self, *paths: str) -> Path:
        return self.get_path("assets", *paths)

    def _create_config(self):
        config = super()._create_config()

        # 初始化下载相关参数
        config.set(
            "DEFAULT_USER_AGENT",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/123.0.0.0 "
            "Safari/537.36"
        )

        return config


environ = Environ()
