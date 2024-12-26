#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import subprocess
from abc import ABC, abstractmethod
from typing import Any, Generator, TypeVar, Callable

from . import utils, Tool, environ
from .decorator import timeoutable
from .types import TimeoutType, Error

BridgeType = TypeVar("BridgeType", bound="Bridge")
DeviceType = TypeVar("DeviceType", bound="BaseDevice")

_logger = environ.get_logger("device")


class BridgeError(Error):
    pass


class Bridge(ABC):

    def __init__(
            self,
            tool: Tool,
            options: [str] = None,
            error_type: Callable[[str], BridgeError] = BridgeError,
            on_stdout: Callable[[str], None] = _logger.info,
            on_stderr: Callable[[str], None] = _logger.error):
        self._tool = tool
        self._options = options or []
        self._error_type = error_type
        self._on_stdout = on_stdout
        self._on_stderr = on_stderr

    @abstractmethod
    def list_devices(self, alive: bool = None) -> Generator["BaseDevice", None, None]:
        """
        获取所有设备列表
        :param alive: 只显示在线的设备
        :return: 设备对象
        """
        pass

    def popen(self, *args: Any, **kwargs) -> utils.Process:
        """
        执行命令
        :param args: 命令参数
        :param kwargs: 其他参数
        :return: 返回进程对象
        """
        return self._tool.popen(
            *(*self._options, *args),
            **kwargs
        )

    @timeoutable
    def exec(self, *args: Any,
             timeout: TimeoutType = None,
             stdout: bool = None, stderr: bool = None,
             log_output: bool = None,
             ignore_errors: bool = False,
             kill_on_return: bool = True) -> str:
        """
        执行命令
        :param args: 命令参数
        :param timeout: 超时时间
        :param stdout: 标准输出，默认为PIPE
        :param stderr: 标准错误，默认为PIPE
        :param log_output: 把输出打印到logger中
        :param ignore_errors: 忽略错误，报错不会抛异常
        :param kill_on_return: 在返回时是否杀掉进程
        :return: 返回输出结果
        """

        if stdout is None:
            stdout = subprocess.PIPE
        if stderr is None:
            stderr = subprocess.PIPE

        process = self.popen(*args, stdout=stdout, stderr=stderr)
        try:
            return self._exec(
                process,
                timeout=timeout,
                log_output=log_output,
                ignore_errors=ignore_errors
            )
        finally:
            if kill_on_return:
                process.kill()

    def _exec(self, process: utils.Process, timeout: TimeoutType, log_output: bool, ignore_errors: bool) -> str:
        out = err = None
        for _out, _err in process.fetch(timeout=timeout):
            if _out is not None:
                out = _out if out is None else out + _out
                if log_output:
                    data: str = _out.decode(errors="ignore") if isinstance(_out, bytes) else _out
                    data = data.rstrip()
                    if data:
                        _logger.info(data)
            if _err is not None:
                err = _err if err is None else err + _err
                if log_output:
                    data: str = _err.decode(errors="ignore") if isinstance(_err, bytes) else _err
                    data = data.rstrip()
                    if data:
                        _logger.error(data)

        if not ignore_errors and process.poll() not in (0, None):
            if isinstance(err, bytes):
                err = err.decode(errors="ignore")
                err = err.strip()
            elif isinstance(err, str):
                err = err.strip()
            if err:
                raise self._error_type(err)

        if isinstance(out, bytes):
            out = out.decode(errors="ignore")
            out = out.strip()
        elif isinstance(out, str):
            out = out.strip()

        return out or ""


class BaseDevice(ABC):

    @property
    @abstractmethod
    def id(self) -> str:
        """
        获取设备号
        :return: 设备号
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        获取设备号
        :return: 设备名
        """
        pass

    @abstractmethod
    def copy(self, timeout: TimeoutType):
        """
        获取设备名
        :param timeout: 超时时间
        :return: 设备名
        """
        pass

    @property
    def pretty_id(self):
        """
        获取可读的设备号信息
        :return: 设备号信息
        """
        name = utils.ignore_error(lambda: f"({self.name})", default="")
        return f"{self.id} {name}" if name else ""


def list_devices(alive: bool = None) -> Generator["BaseDevice", None, None]:
    """
    获取所有设备列表（包括Android、iOS、Harmony）
    :param alive: 只显示在线的设备
    :return: 设备对象
    """
    from .android import Adb
    from .ios import Sib
    from .harmony import Hdc
    for device in Adb().list_devices(alive=alive):
        yield device
    for device in Sib().list_devices(alive=alive):
        yield device
    for device in Hdc().list_devices(alive=alive):
        yield device
