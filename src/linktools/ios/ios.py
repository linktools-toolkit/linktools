#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : ios.py
@time    : 2024/10/14 16:25 
@site    : https://github.com/ice-black-tea
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
import json
import subprocess
import time
from typing import TYPE_CHECKING, TypeVar, List, Generator, Any, Callable, Dict

from .struct import App, Process
from .. import utils, environ
from ..decorator import timeoutable, cached_property
from ..device import BridgeError, Bridge, BaseDevice
from ..types import Stoppable, TimeoutType, Timeout

if TYPE_CHECKING:
    from ..ssh import SSHClient

    DEVICE_TYPE = TypeVar("DEVICE_TYPE", bound="GoIOSDevice")

_logger = environ.get_logger("ios.go-ios")


def _is_log(data: Any) -> bool:
    return isinstance(data, dict) and "level" in data and "msg" in data


def _get_msg(log: "Dict[str, str]") -> str:
    return log.get("msg")


def _get_err_msg(log: "Dict[str, str]") -> str:
    return log.get("error") or log.get("msg")


class GoIOSError(BridgeError):
    pass


class GoIOS(Bridge):

    def __init__(self, options: List[str] = None):
        super().__init__(
            tool=environ.get_tool("ios"),
            options=options,
            error_type=GoIOSError,
        )

    def list_devices(self, alive: bool = None) -> Generator["GoIOSDevice", None, None]:
        """
        获取所有设备列表
        :param alive: 只显示在线的设备
        :return: 设备号数组
        """
        result = self.exec("list")
        for line in result.splitlines():
            data = utils.ignore_error(json.loads, args=(line,), default=None)
            if isinstance(data, dict) and "deviceList" in data:
                for id in data["deviceList"]:
                    if alive is None:
                        yield GoIOSDevice(id, ios=self)
                    elif alive is True:  # online only
                        yield GoIOSDevice(id, ios=self)

    def _exec(self, process: utils.Process, timeout: TimeoutType, log_output: bool, ignore_errors: bool) -> str:
        result = None
        for out, err in process.fetch(timeout=timeout):
            if out is not None:
                result = out if result is None else result + out
            for line in (out or "").splitlines() + (err or "").splitlines():
                data: dict = utils.ignore_error(json.loads, args=(line,), default=None)
                if log_output:
                    if _is_log(data):
                        level = data.get("level")
                        if level in ("error", "fatal"):
                            _logger.error(_get_err_msg(data))
                        elif level in ("warning",):
                            _logger.warning(_get_msg(data))
                        elif level in ("trace", "debug"):
                            _logger.debug(_get_msg(data))
                        else:
                            _logger.info(_get_msg(data))
                    elif data:
                        _logger.info(data)
                if not ignore_errors:
                    if _is_log(data):
                        level = data.get("level")
                        if level in ("error", "fatal"):
                            if not ignore_errors:
                                raise self._error_type(_get_err_msg(data))

        if isinstance(result, bytes):
            result = result.decode(errors="ignore")
            result = result.strip()
        elif isinstance(result, str):
            result = result.strip()

        return result or ""


class GoIOSDevice(BaseDevice):

    def __init__(self, id: str = None, ios: GoIOS = None):
        """
        :param id: 设备号
        :param ios: IOS对象
        """
        self._ios = ios or GoIOS()
        if id is None:
            devices = list(self._ios.list_devices(alive=True))
            if len(devices) == 0:
                raise GoIOSError("no devices/emulators found")
            elif len(devices) > 1:
                raise GoIOSError("more than one device/emulator")
            self._id = devices[0]._id
        else:
            self._id = id

    @property
    def id(self) -> str:
        """
        获取设备号
        :return: 设备号
        """
        return self._id

    @property
    def name(self) -> str:
        """
        获取设备名称
        :return: 设备名称
        """
        return self.info.get("DeviceName")

    @property
    def version(self) -> str:
        """
        获取系统版本
        :return: 系统版本
        """
        return self.info.get("ProductVersion")

    @property
    def type(self) -> str:
        """
        获取设备型号
        :return: 设备型号
        """
        return self.info.get("ProductType")

    @cached_property
    def info(self) -> dict:
        """
        获取设备详细信息
        :return: 设备类型
        """
        for line in self.exec("info").splitlines():
            return utils.ignore_error(json.loads, args=(line,), default={})
        raise GoIOSError("get device info failed")

    def copy(self, type: "Callable[[str, GoIOS], DEVICE_TYPE]" = None) -> "DEVICE_TYPE":
        """
        生成一个新的设备对象
        :param type: 设备类型
        :return: 新的设备对象
        """
        return (type or GoIOSDevice)(self._id, self._ios)

    def popen(self, *args: Any, **kwargs) -> utils.Process:
        """
        执行命令
        :param args: 命令行参数
        :return: 打开的进程
        """
        args = ["--udid", self.id, *args]
        return self._ios.popen(*args, **kwargs)

    @timeoutable
    def exec(self, *args: Any, **kwargs) -> str:
        """
        执行命令
        :param args: 命令行参数
        :return: ios输出结果
        """
        args = ["--udid", self.id, *args]
        return self._ios.exec(*args, **kwargs)

    @timeoutable
    def mount(self, **kwargs) -> None:
        """
        挂载image
        """
        path = environ.get_data_path("ios", "image", create_parent=True)
        self.exec("image", "auto", f"--basedir={path}", **kwargs)

    @timeoutable
    def install(self, path_or_url: str, **kwargs) -> str:
        """
        安装应用
        :param path_or_url: 本地路径或者url
        :return: ios输出结果
        """
        _logger.info(f"Install ipa url: {path_or_url}")
        ipa_path = environ.get_url_file(path_or_url).save()
        _logger.debug(f"Local ipa path: {ipa_path}")
        return self.exec("install", f"--path={ipa_path}", **kwargs)

    @timeoutable
    def uninstall(self, bundle_id: str, **kwargs) -> str:
        """
        卸载应用
        :param bundle_id: 包名
        :return: ios输出结果
        """
        return self.exec("uninstall", bundle_id, **kwargs)

    @timeoutable
    def kill(self, bundle_id: str, **kwargs) -> str:
        """
        结束应用
        :param bundle_id: 包名
        :return: ios输出结果
        """
        return self.exec("kill", bundle_id, **kwargs)

    @timeoutable
    def get_app(self, bundle_id: str, **kwargs) -> App:
        """
        根据包名获取包信息
        :param bundle_id: 包名
        :return: 包信息
        """
        for line in self.exec("apps", "--all", **kwargs).splitlines():
            for obj in utils.ignore_error(json.loads, args=(line,), default=[]):
                app = App(obj)
                if bundle_id == app.bundle_id:
                    return app

        raise GoIOSError(f"App '{bundle_id}' not found")

    @timeoutable
    def get_apps(self, *bundle_ids: str, system: bool = None, **kwargs) -> "List[App]":
        """
        获取包信息
        :param bundle_ids: 需要匹配的所有包名，为空则匹配所有
        :param system: true只匹配系统应用，false只匹配非系统应用，为空则全匹配
        :return: 包信息
        """
        options = []
        if system is None:
            options.append("--all")
        elif system is True:
            options.append("--system")

        result = []
        for line in self.exec("apps", *options, **kwargs).splitlines():
            for obj in utils.ignore_error(json.loads, args=(line,), default=[]):
                app = App(obj)
                if not bundle_ids or app.bundle_id in bundle_ids:
                    result.append(app)

        return result

    @timeoutable
    def get_processes(self, **kwargs) -> "List[Process]":
        """
        获取进程列表
        :return: 进程列表
        """
        result = []
        for line in self.exec("ps", **kwargs).splitlines():
            for obj in utils.ignore_error(json.loads, args=(line,), default=[]):
                result.append(Process(obj))
        return result

    def forward(self, local_port: int, remote_port: int) -> "Forward":
        """
        创建端口转发
        :param local_port: 本地端口
        :param remote_port: 远程端口
        :return: 端口转发对象
        """
        return Forward(self, local_port, remote_port)

    def ssh(self, port: int = 22, username: str = "root", password: str = None) -> "SSHClient":
        """
        创建ssh连接，需要ios设备已完成越狱
        :param port: ssh端口
        :param username: 用户名
        :param password: 密码
        :return: ssh连接
        """
        import paramiko
        from linktools.ssh import SSHClient

        forward = None
        client = None
        try:
            forward = self.forward(
                local_port=utils.get_free_port(),
                remote_port=port,
            )

            class Client(SSHClient):

                def close(self):
                    try:
                        super().close()
                    finally:
                        forward.stop()

            client = Client()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            client.connect_with_pwd(
                "localhost",
                port=forward.local_port,
                username=username,
                password=password,
            )
        except:
            if client is not None:
                utils.ignore_error(client.close)
            elif forward is not None:
                forward.stop()
            raise

        return client

    def __repr__(self):
        return f"IOSDevice<{self.id}>"


class Forward(Stoppable):
    local_port = property(lambda self: self._local_port)
    remote_port = property(lambda self: self._remote_port)

    def __init__(self, ios: GoIOSDevice, local_port: int, remote_port: int):
        self._local_port = local_port
        self._remote_port = remote_port
        self._process = None

        def start():
            self._process = ios.popen(
                "forward",
                local_port,
                remote_port,
                text=True,
                bufsize=1,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for i in range(5):
                timeout = Timeout(min(i, 2))
                for out, err in self._process.fetch(timeout=timeout):
                    for line in (out or "").splitlines() + (err or "").splitlines():
                        data = utils.ignore_error(json.loads, args=(line,), default=None)
                        if _is_log(data):
                            level = data.get("level")
                            if level in ("fatal", "error"):
                                raise GoIOSError(_get_err_msg(data))
                            elif "Start listening on port" in data["msg"]:
                                _logger.debug(f"Capture ios {self} output: {data['msg']}")
                                time.sleep(1)
                                break

                if self._process.poll() is None and not utils.is_port_free(local_port):
                    _logger.debug(f"{self} port {local_port} is used, continue.")
                    return

                _logger.debug(f"Start forward failed, kill {self} process and restart it.")
                utils.ignore_error(self._process.recursive_kill)
                utils.wait_process(self._process, .5)
                self._process = ios.popen(
                    "forward",
                    local_port,
                    remote_port,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

            raise GoIOSError("Run ios forward failed")

        self._stop_on_error(start)

    def stop(self):
        process, self._process = self._process, None
        if process is not None:
            _logger.debug(f"Kill {self} process")
            utils.ignore_error(process.recursive_kill)
            utils.wait_process(process, 5)

    def __repr__(self):
        return f"Forward<{self.local_port}:{self.remote_port}>"


if __name__ == '__main__':
    import logging
    from .. import rich

    rich.init_logging(level=logging.DEBUG, show_level=True)

    device = GoIOSDevice()
    print(device.mount(log_output=True))
    print(device.get_apps(system=False))
    print(device.get_processes())
    # print(device.exec("ps", log_output=True))
    # with device.ssh(22, "root", "alpine") as ssh:
    #     ssh.open_shell()
