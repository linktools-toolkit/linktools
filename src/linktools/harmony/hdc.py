#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import Any, Generator, List, Callable, TYPE_CHECKING, TypeVar

from .. import utils
from .._environ import environ
from ..decorator import cached_property, timeoutable
from ..device import BridgeError, Bridge, BaseDevice
from ..types import TimeoutType

if TYPE_CHECKING:
    DEVICE_TYPE = TypeVar("DEVICE_TYPE", bound="Device")

_logger = environ.get_logger("harmony.hdc")


class HdcError(BridgeError):
    pass


class Hdc(Bridge):

    def __init__(self, options: List[str] = None):
        super().__init__(
            tool=environ.get_tool("hdc"),
            options=options,
            error_type=HdcError
        )

    def list_devices(self, alive: bool = None) -> Generator["HdcDevice", None, None]:
        """
        获取所有设备列表
        :param alive: 只显示在线的设备
        :return: 设备号数组
        """
        result = self.exec("list", "targets", "-v")
        for line in result.splitlines():
            splits = line.split()
            if len(splits) == 4:
                id, mode, status, address = splits[0], splits[1], splits[2], splits[3]
                if alive is None:
                    yield HdcDevice(id)
                elif alive == (status in ("Connected",)):
                    yield HdcDevice(id)


class HdcDevice(BaseDevice):

    def __init__(self, id: str = None, hdc: Hdc = None):
        """
        :param id: 设备号
        """
        self._hdc = hdc or Hdc()
        if id is None:
            devices = list(self._hdc.list_devices(alive=True))
            if len(devices) == 0:
                raise HdcError("no devices/emulators found")
            elif len(devices) > 1:
                raise HdcError("more than one device/emulator")
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
        return self.get_prop("const.product.model", timeout=1)

    @cached_property
    def abi(self) -> str:
        """
        获取设备abi类型
        :return: abi类型
        """
        result = self.get_prop("const.product.cpu.abilist")
        if result.find("arm64") >= 0:
            return "arm64"
        elif result.find("armeabi") >= 0:
            return "arm"
        elif result.find("x86_64") >= 0:
            return "x86_64"
        elif result.find("x86") >= 0:
            return "x86"
        raise HdcError("unknown abi: %s" % result)

    @cached_property
    def uid(self) -> int:
        """
        获取shell的uid
        :return: uid
        """
        return self.get_uid()

    def copy(self, type: "Callable[[str, Hdc], DEVICE_TYPE]" = None) -> "DEVICE_TYPE":
        """
        生成一个新的设备对象
        :param type: 设备类型
        :return: 新的设备对象
        """
        return (type or HdcDevice)(self._id, self._hdc)

    @timeoutable
    def exec(self, *args: Any, **kwargs) -> str:
        """
        执行命令
        :param args: 命令行参数
        :return: hdc输出结果
        """
        args = ["-t", self.id, *args]
        return self._hdc.exec(*args, **kwargs)

    def make_shell_args(self, *args: Any):
        """
        生成shell命令参数
        :param args: shell命令
        :return: hdc命令参数
        """
        cmd = utils.list2cmdline([str(arg) for arg in args])
        return ["shell", cmd]

    @timeoutable
    def shell(self, *args: Any, **kwargs) -> str:
        """
        执行shell
        :param args: shell命令
        :return: hdc输出结果
        """
        args = self.make_shell_args(*args)
        return self.exec(*args, **kwargs)

    @timeoutable
    def get_prop(self, prop: str, **kwargs) -> str:
        """
        获取属性值
        :param prop: 属性名
        :return: 属性值
        """
        return self.shell("param", "get", prop, **kwargs).rstrip()

    @timeoutable
    def get_uid(self, timeout: TimeoutType = None) -> int:
        out = self.shell("id", "-u", timeout=timeout)
        uid = utils.int(out.strip(), default=-1)
        if uid >= 0:
            return uid
        out = self.shell("echo", "-n", "${USER_ID}", timeout=timeout)
        uid = utils.int(out.strip(), default=-1)
        if uid >= 0:
            return uid
        raise HdcError("unknown hdc uid: %s" % out)

    def __repr__(self):
        return f"HdcDevice<{self.id}>"
