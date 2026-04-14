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
from typing import TYPE_CHECKING, Type, Any

from linktools import utils, metadata
from linktools.decorator import cached_property, cached_classproperty

if TYPE_CHECKING:
    from linktools.types import T, ConfigDict, Config, Tools, Tool, UrlFile, PathType


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
        return utils.get_system()

    @property
    def machine(self) -> str:
        """Return the machine architecture.

        Returns:
            str: The property value.
        """
        return utils.get_machine()

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
    def data_path(self) -> Path:
        """Data path.

        Returns:
            Path: The operation result.
        """
        return Path(self.global_config["DATA_PATH"])

    @cached_property
    def temp_path(self) -> Path:
        """Temp path.

        Returns:
            Path: The operation result.
        """
        return Path(self.global_config["TEMP_PATH"])

    def get_path(self, *paths: str) -> Path:
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

    def get_data_path(self, *paths: str, create_parent: bool = False) -> Path:
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

    def get_temp_path(self, *paths: str, create_parent: bool = False) -> Path:
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
        return logging.getLogger(self.name)

    def get_logger(self, name: str = None) -> "logging.Logger":
        """Return the logger.

        Args:
            name (str): Name to resolve.

        Returns:
            logging.Logger: The operation result.
        """
        name = f"{self.name}.{name}" if name else self.name
        return logging.getLogger(name)

    @cached_classproperty(lock=True)
    def global_config(self) -> "ConfigDict":
        """Build the global configuration dictionary.

        Returns:
            ConfigDict: The operation result.
        """
        from ._config import ConfigDict

        prefix = f"{metadata.__name__}".upper()

        data_path = os.environ.get(f"{prefix}_DATA_PATH", None)
        temp_path = os.environ.get(f"{prefix}_TEMP_PATH", None)
        if not (data_path and temp_path):
            storage_path = os.environ.get(f"{prefix}_PATH", None)
            if not storage_path:
                storage_path = os.environ.get(f"{prefix}_STORAGE_PATH", None)
                if not storage_path:
                    storage_path = os.path.join(Path.home(), f".{metadata.__name__}")
            if not data_path:
                data_path = os.path.join(storage_path, "data")
            if not temp_path:
                temp_path = os.path.join(storage_path, "temp")

        return ConfigDict(
            DEBUG=False,
            DATA_PATH=data_path,
            TEMP_PATH=temp_path,
        )

    def _create_config(self) -> "Config":
        from ._config import Config, ConfigDict

        return Config(
            self,
            ConfigDict(),
            namespace="MAIN",
            env_prefix=f"{self.name.upper()}_"
        )

    @cached_property(lock=True)
    def config(self) -> "Config":
        """Config.

        Returns:
            Config: The operation result.
        """
        return self._create_config()

    def wrap_config(self, namespace: str = metadata.__missing__, env_prefix: str = metadata.__missing__) -> "Config":
        """Return a scoped configuration wrapper.

        Args:
            namespace (str): Argparse namespace to update.
            env_prefix (str): The env_prefix value.

        Returns:
            Config: The operation result.
        """
        from ._config import ConfigWrapper

        return ConfigWrapper(self.config, namespace=namespace, env_prefix=env_prefix)

    def get_config(self, key: str, type: "Type[T]" = None, default: Any = metadata.__missing__) -> "T":
        """Return a configuration value.

        Args:
            key (str): Configuration or item key.
            type (Type[T]): Target type used to cast the value.
            default (Any): Value returned when no explicit value is available.

        Returns:
            T: The operation result.
        """
        return self.config.get(key=key, type=type, default=default)

    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value.

        Args:
            key (str): Configuration or item key.
            value (Any): Value to store or process.
        """
        self.config.set(key, value)

    def _create_tools(self) -> "Tools":
        from ._tools import Tools
        from ._config import ConfigDict

        config = ConfigDict()

        develop_path = environ.get_path("assets", "develop", "tools.yml")
        data_path = environ.get_data_path("tools", "tools.json")
        asset_path = environ.get_path("assets", "tools.json")

        if os.path.exists(data_path):
            config.update_from_file(data_path, json.load)
        elif not metadata.__develop__ or not os.path.exists(develop_path):
            config.update_from_file(asset_path, json.load)
        else:
            import yaml
            config.update_from_file(develop_path, yaml.safe_load)

        tools = Tools(self, config)
        paths = os.environ["PATH"].split(os.pathsep)
        stub_path = str(tools.stub_path)
        if stub_path not in paths:
            paths.append(stub_path)
            os.environ["PATH"] = os.pathsep.join(paths)
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
        from ._url import HttpFile, LocalFile

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
    def root_path(self) -> Path:
        """Return the root directory for the current package.

        Returns:
            Path: The operation result.
        """
        return Path(os.path.dirname(os.path.dirname(__file__)))

    def _create_config(self):
        config = super()._create_config()

        # Initialize download-related defaults.
        config.update(
            DEFAULT_USER_AGENT=
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/140.0.0.0 "
            "Safari/537.36",
            DEFAULT_WAN_IP_URL="http://ifconfig.me/ip"  # noqa
        )

        return config


environ = Environ()
