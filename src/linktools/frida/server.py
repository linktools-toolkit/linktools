#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : frida.py
@time    : 2021/12/18
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
import gzip
import json
import lzma
import os
import shutil
import time
from typing import List, Dict, Optional

import frida

from .. import environ, utils, DownloadHttpError
from ..android import AdbDevice
from ..decorator import cached_classproperty
from ..ios import GoIOSDevice
from ..types import Timeout, Stoppable

_logger = environ.get_logger("frida.server")


class FridaServer(utils.get_derived_type(frida.core.Device), Stoppable, metaclass=abc.ABCMeta):  # proxy for frida.core.Device

    def __init__(self, device: frida.core.Device):
        super().__init__(device)

    @property
    def is_running(self) -> bool:
        """
        判断服务端运行状态
        :return: 是否正在运行
        """
        try:
            processes = self.enumerate_processes()
            return processes is not None
        except (frida.TransportError, frida.ServerNotRunningError, frida.InvalidArgumentError) as e:
            _logger.debug(f"Frida server is not running: {e}")
            return False

    def start(self) -> bool:
        """
        根据frida版本和设备abi类型下载并运行server
        :return: 运行成功为True，否则为False
        """
        try:

            if self.is_running:
                _logger.info("Frida server is running ...")
                return True

            _logger.info("Start frida server ...")
            self._start()

            timeout = Timeout(10)
            while timeout.check():
                if self.is_running:
                    _logger.info("Frida server is running ...")
                    return True
                time.sleep(min(timeout.remain, 0.5))

            raise frida.ServerNotRunningError("Frida server failed to run ...")

        except BaseException as e:
            _logger.debug("Kill frida server ...")
            utils.ignore_error(self._stop)
            raise e

    def stop(self) -> bool:
        """
        强制结束frida server
        :return: 结束成功为True，否则为False
        """
        _logger.info("Kill frida server ...")
        try:
            self._stop()
            return True
        except frida.ServerNotRunningError:
            return True
        except:
            return False

    @abc.abstractmethod
    def _start(self):
        pass

    @abc.abstractmethod
    def _stop(self):
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class FridaAndroidServer(FridaServer):
    """
    android server
    """

    def __init__(self, device: AdbDevice = None, local_port: int = 47042, remote_port: int = 47042, serve: bool = True):
        super().__init__(frida.get_device_manager().add_remote_device(f"localhost:{local_port}"))
        self._device = device or AdbDevice()
        self._local_port = local_port
        self._remote_port = remote_port
        self._forward: Optional[Stoppable] = None

        self._serve = serve
        self._server_name = f"fs-ln-{self._remote_port}"
        self._server_dir = self._device.get_data_path("fs-ln")
        self._server_path = self._device.join_path(self._server_dir, self._server_name)

    @property
    def local_port(self):
        return self._local_port

    @property
    def remote_port(self):
        return self._remote_port

    def _start(self):
        # 转发端口
        if self._forward:
            self._forward.stop()

        self._forward = self._device.forward(
            f"tcp:{self._local_port}",
            f"tcp:{self._remote_port}"
        )

        if self._serve:
            # 创建软链
            server_path = self._prepare_executable()
            self._device.sudo("mkdir", "-p", self._server_dir)
            self._device.sudo("ln", "-sf", server_path, self._server_path)

            # 接下来新开一个进程运行frida server，并且输出一下是否出错
            self._device.sudo(
                self._server_path,
                "-d", "fs-binaries",
                "-l", f"0.0.0.0:{self._remote_port}",
                "-D", "&",
                ignore_errors=True,
                log_output=True,
            )

    def _stop(self):
        try:
            if self._serve:
                # 就算杀死adb进程，frida server也不一定真的结束了，所以kill一下frida server进程
                self._device.sudo("killall", self._server_name, ignore_errors=True)
        finally:
            if self._forward:
                # 把转发端口给移除了，不然会一直占用这个端口
                self._forward.stop()
                self._forward = None

    @cached_classproperty
    def _server_info(self) -> "List[Dict[str, str]]":
        server_path = environ.get_asset_path("android-tools.json")
        server_data = json.loads(utils.read_file(server_path, text=True))
        return server_data["FRIDA_SERVER"]

    @classmethod
    def _get_executables(cls, abi: str, version: str):
        result = []
        for config in cls._server_info:
            config = dict(config)
            config.update(version=version, abi=abi)
            min_version = config.get("min_version", "0.0.0")
            max_version = config.get("max_version", "99999.0.0")
            if utils.parse_version(min_version) <= utils.parse_version(version) <= utils.parse_version(max_version):
                result.append(cls.Executable(config))
        return result

    def _prepare_executable(self) -> str:
        executables = self._get_executables(self._device.abi, frida.__version__)

        # 先判断设备上有没有现成的frida server，有的话直接返回
        for executable in executables:
            remote_path = self._device.get_data_path("fs", executable.name)
            if self._device.is_file_exist(remote_path):
                return remote_path

        # 设备上如果没有，那需要下载了，默认按照配置里的顺序进行下载
        for executable in executables:
            remote_dir = self._device.get_data_path("fs")
            remote_path = self._device.get_data_path("fs", executable.name)

            # 先下载frida server，然后把server推送到设备上
            try:
                executable.download()
            except DownloadHttpError as e:
                if 400 <= e.code < 500:
                    continue
                raise e

            _logger.info(f"Push {executable.name} to remote: {remote_path}")

            temp_dir = self._device.get_data_path("temp")
            temp_path = self._device.push_file(executable.path, temp_dir, log_output=True)
            self._device.sudo("mkdir", "-p", remote_dir, log_output=True)
            self._device.sudo("mv", temp_path, remote_path, log_output=True)
            self._device.sudo("chmod", "755", remote_path, log_output=True)

            return remote_path

        raise FileNotFoundError("Frida server not found ...")

    @classmethod
    def setup(cls, abis: [str] = ("arm", "arm64", "x86_64", "x86"), version: str = frida.__version__):
        for abi in abis:
            for executable in cls._get_executables(abi, version):
                try:
                    executable.download()
                except DownloadHttpError as e:
                    if 400 <= e.code < 500:
                        continue
                    raise e
                break

    class Executable:

        def __init__(self, config: Dict[str, str]):
            self.url = config["url"].format(**config)
            self.name = config["name"].format(**config)
            self.path = environ.get_data_path("android", "frida", self.name, create_parent=True)

        def download(self):
            if os.path.exists(self.path):
                return
            _logger.info("Download frida server ...")
            with environ.get_url_file(self.url) as file:
                if os.path.exists(self.path):
                    return
                temp_path = file.save()
                temp_name = utils.guess_file_name(self.url)
                if temp_name.endswith(".xz"):
                    with lzma.open(temp_path, "rb") as read, open(self.path, "wb") as write:
                        shutil.copyfileobj(read, write)
                elif temp_name.endswith(".gz"):
                    with gzip.GzipFile(temp_path, "rb") as read, open(self.path, "wb") as write:
                        shutil.copyfileobj(read, write)

                file.clear()


class FridaIOSServer(FridaServer):  # proxy for frida.core.Device
    """
    ios server
    """

    def __init__(self, device: GoIOSDevice = None, local_port: int = 37042, remote_port: int = 27042):
        super().__init__(frida.get_device_manager().add_remote_device(f"localhost:{local_port}"))
        self._device = device or GoIOSDevice()
        self._local_port = local_port
        self._remote_port = remote_port
        self._forward: Optional[Stoppable] = None

    @property
    def local_port(self):
        return self._local_port

    @property
    def remote_port(self):
        return self._remote_port

    def _start(self):
        if self._forward:
            self._forward.stop()

        self._forward = self._device.forward(
            self._local_port,
            self._remote_port
        )

    def _stop(self):
        if self._forward:
            utils.ignore_error(self._forward.stop)
            self._forward = None
