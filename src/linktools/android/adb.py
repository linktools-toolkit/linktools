#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : adb.py
@time    : 2018/11/25
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

import json
import os
import re
import time
from typing import Any, Generator, List, Callable, TYPE_CHECKING, TypeVar

from .struct import App, UnixSocket, InetSocket, Process, File, SystemService
from .. import utils
from .._environ import environ
from ..decorator import cached_property, cached_classproperty, timeoutable
from ..device import BridgeError, Bridge, BaseDevice
from ..types import TimeoutType, Stoppable

if TYPE_CHECKING:
    DEVICE_TYPE = TypeVar("DEVICE_TYPE", bound="AdbDevice")

_logger = environ.get_logger("android.adb")
_agent_output_pattern = re.compile(
    r"┌──+──┐[^\n]*\n"
    r"│[^|]*│[^\n]*\n"
    r"└──+──┘[^\n]*\n?",
    re.MULTILINE
)


class AdbError(BridgeError):
    pass


class Adb(Bridge):

    def __init__(self, options: List[str] = None):
        super().__init__(
            tool=environ.get_tool("adb"),
            options=options,
            error_type=AdbError
        )

    def list_devices(self, alive: bool = None) -> Generator["AdbDevice", None, None]:
        """
        获取所有设备列表
        :param alive: 只显示在线的设备
        :return: 设备号数组
        """
        result = self.exec("devices")
        for line in result.splitlines()[1:]:
            splits = line.split(maxsplit=1)
            if len(splits) >= 2:
                device, status = splits
                if alive is None:
                    yield AdbDevice(device, adb=self)
                elif alive == (status in ("bootloader", "device", "recovery", "sideload")):
                    yield AdbDevice(device, adb=self)


class AdbDevice(BaseDevice):

    def __init__(self, id: str = None, adb: Adb = None):
        """
        :param id: 设备号
        """
        self._adb = adb or Adb()
        if id is None:
            devices = tuple(self._adb.list_devices(alive=True))
            if len(devices) == 0:
                raise AdbError("no devices/emulators found")
            elif len(devices) > 1:
                raise AdbError("more than one device/emulator")
            self._id = devices[0]._id
        else:
            self._id = id

    @cached_property
    def id(self) -> str:
        """
        获取设备号
        :return: 设备号
        """
        return self._id

    @cached_property
    def name(self) -> str:
        """
        获取设备名
        :return: 设备名
        :raise AdbError: 获取设备名失败
        """
        return self.get_prop("ro.product.model", timeout=1)

    @cached_property
    def abi(self) -> str:
        """
        获取设备abi类型
        :return: abi类型
        :raise AdbError: 获取abi类型失败
        """
        result = self.get_prop("ro.product.cpu.abi")
        if result.find("arm64") >= 0:
            return "arm64"
        elif result.find("armeabi") >= 0:
            return "arm"
        elif result.find("x86_64") >= 0:
            return "x86_64"
        elif result.find("x86") >= 0:
            return "x86"
        raise AdbError("unknown abi: %s" % result)

    @cached_property
    def uid(self) -> int:
        """
        获取shell的uid
        :return: uid
        """
        return self.get_uid()

    def copy(self, type: "Callable[[str, Adb], DEVICE_TYPE]" = None) -> "DEVICE_TYPE":
        """
        生成一个新的设备对象
        :param type: 设备类型
        :return: 新的设备对象
        """
        return (type or AdbDevice)(self._id, self._adb)

    def popen(self, *args: Any, **kwargs) -> utils.Process:
        """
        执行命令
        :param args: 命令行参数
        :return: 打开的进程
        """
        args = ["-s", self.id, *args]
        return self._adb.popen(*args, **kwargs)

    @timeoutable
    def exec(self, *args: Any, **kwargs) -> str:
        """
        执行命令
        :param args: 命令行参数
        :return: adb输出结果
        """
        args = ["-s", self.id, *args]
        return self._adb.exec(*args, **kwargs)

    def make_shell_args(self, *args: Any, privilege: bool = False, user: str = None):
        """
        生成shell参数
        :param args: 参数
        :param privilege: 是否以root权限运行
        :param user: 以指定user运行
        :return: 参数列表
        """
        cmd = utils.list2cmdline([str(arg) for arg in args])
        if privilege and self.uid != 0:
            args = ["shell", "su", "-c", cmd]
        elif user:
            args = ["shell", "su", user, "-c", cmd]
        else:
            args = ["shell", cmd]
        return args

    @timeoutable
    def shell(self, *args: Any, privilege: bool = False, user: str = None, **kwargs) -> str:
        """
        执行shell
        :param args: shell命令
        :param privilege: 是否以root权限运行
        :param user: 以指定user运行
        :return: adb输出结果
        """
        args = self.make_shell_args(*args, privilege=privilege, user=user)
        return self.exec(*args, **kwargs)

    @timeoutable
    def sudo(self, *args: Any, **kwargs) -> str:
        """
        以root权限执行shell
        :param args: shell命令
        :return: adb输出结果
        """
        kwargs["privilege"] = True
        return self.shell(*args, **kwargs)

    @timeoutable
    def install(self, path_or_url: str, opts: [str] = (), **kwargs):
        """
        安装apk
        :param path_or_url: apk文件路径
        :param opts: 安装参数
        """
        _logger.info(f"Install apk url: {path_or_url}")
        apk_path = environ.get_url_file(path_or_url).save()
        _logger.debug(f"Local apk path: {apk_path}")

        remote_name = f"installed_{int(time.time())}.apk"
        remote_path = self.push_file(apk_path, self.get_data_path("android", "apk"), remote_name, **kwargs)
        try:
            if self.uid >= 10000:
                self.shell("am", "start", "--user", "0",
                           "-a", "android.intent.action.VIEW",
                           "-t", "application/vnd.android.package-archive",
                           "-d", "file://%s" % remote_path,
                           **kwargs)
            else:
                self.shell("pm", "install", *(opts or tuple()), remote_path,
                           **kwargs)
        finally:
            self.shell("rm", remote_path, **kwargs, ignore_errors=True)

    @timeoutable
    def uninstall(self, package_name: str, **kwargs):
        """
        卸载apk
        :param package_name: 包名
        :return: adb输出结果
        """
        self.exec("uninstall", package_name, **kwargs)

    @timeoutable
    def push_file(self, src_path: str, dest_dir: str, dest_name: str = None, skip_exist: bool = False, **kwargs) -> str:
        """
        推送文件到设备
        :param src_path: 源文件（本地）
        :param dest_dir: 目标文件夹（Android设备）
        :param dest_name: 目标文件名（Android设备）
        :param skip_exist: 跳过已存在的目标文件
        :return: 目标文件路径
        """
        ignore_errors = kwargs.get("ignore_errors", False)
        dest_name = dest_name or os.path.basename(src_path)
        dest_path = self.join_path(dest_dir, dest_name)
        if not ignore_errors and not os.path.exists(src_path):
            raise AdbError(f"{src_path} does not exist in host")
        if not skip_exist or not self.is_file_exist(dest_path, **kwargs):
            self.shell("mkdir", "-p", dest_dir, **kwargs)
            self.exec("push", src_path, dest_path, **kwargs)
        return dest_path

    def push_dir(self, src_dir: str, dest_dir: str, skip_exist: bool = False, **kwargs) -> str:
        """
        推送文件夹到设备
        :param src_dir: 源文件夹（本地）
        :param dest_dir: 目标文件夹（Android设备）
        :param skip_exist: 跳过已存在的目标文件
        :return: adb输出结果
        """
        ignore_errors = kwargs.get("ignore_errors", False)
        src_dir = os.path.abspath(os.path.expanduser(src_dir))
        if not ignore_errors and not os.path.exists(src_dir):
            raise AdbError(f"{src_dir} does not exist in host")
        if not ignore_errors and not os.path.isdir(src_dir):
            raise AdbError(f"{src_dir} is not a directory in host")
        if not skip_exist or not self.is_directory_exist(dest_dir, **kwargs):
            self.shell("mkdir", "-p", dest_dir, **kwargs)
            self.exec("push", os.path.join(src_dir, "."), dest_dir, **kwargs)
        return dest_dir

    @timeoutable
    def pull_file(self, src_path: str, dest_dir: str, dest_name: str = None, skip_exist: bool = False, **kwargs) -> str:
        """
        从设备拉取文件
        :param src_path: 源文件（Android设备）
        :param dest_dir: 目标文件夹（本地）
        :param dest_name: 目标文件名（本地）
        :param skip_exist: 跳过已存在的目标文件
        :return: adb输出结果
        """
        ignore_errors = kwargs.get("ignore_errors")
        dest_name = dest_name or self.get_base_name(src_path)
        dest_path = os.path.join(dest_dir, dest_name)
        if not ignore_errors and not self.is_file_exist(src_path):
            raise AdbError(f"{src_path} does not exist in {self}")
        is_exist = os.path.exists(dest_path)
        if not skip_exist or not is_exist:
            if not is_exist:
                os.makedirs(dest_dir, exist_ok=True)
            self.exec("pull", src_path, dest_path, **kwargs)
        return dest_path

    @timeoutable
    def pull_dir(self, src_dir: str, dest_dir: str, skip_exist: bool = False, **kwargs) -> str:
        """
        从设备拉取文件夹
        :param src_dir: 源文件（Android设备）
        :param dest_dir: 目标文件夹（本地）
        :param skip_exist: 跳过已存在的目标文件
        :return: adb输出结果
        """
        ignore_errors = kwargs.get("ignore_errors")
        dest_dir = os.path.abspath(os.path.expanduser(dest_dir))
        if not ignore_errors and not self.is_directory_exist(src_dir):
            raise AdbError(f"{src_dir} does not exist in {self}")
        is_exist = os.path.exists(dest_dir)
        if not skip_exist or not is_exist:
            if not is_exist:
                os.makedirs(dest_dir, exist_ok=True)
            self.exec("pull", self.join_path(src_dir, "."), dest_dir, **kwargs)
        return dest_dir

    def forward(self, local: str, remote: str) -> "AdbForward":
        """
        端口转发
        :param local: 本地端口
        :param remote: 远程端口
        :return: 可关闭对象
        """
        return AdbForward(self, local, remote)

    def reverse(self, remote: str, local: str) -> "AdbReverse":
        """
        端口转发
        :param remote: 远程端口
        :param local: 本地端口
        :return: 可关闭对象
        """
        return AdbReverse(self, remote, local)

    def redirect(self, address: str = None, port: int = None, uid: int = None) -> "AdbRedirect":
        """
        将手机流量重定向到本地指定端口
        :param address: 本地监听地址，不填默认本机
        :param port: 本地监听端口
        :param uid: 监听目标uid
        :return: 重定向对象
        """
        if not port:
            port = utils.get_free_port()

        return AdbRedirect(self, address, port, uid)

    @timeoutable
    def get_prop(self, prop: str, **kwargs) -> str:
        """
        获取属性值
        :param prop: 属性名
        :return: 属性值
        """
        return self.shell("getprop", prop, **kwargs).rstrip()

    @timeoutable
    def set_prop(self, prop: str, value: str, **kwargs) -> str:
        """
        设置属性值
        :param prop: 属性名
        :param value: 属性值
        :return: adb输出结果
        """
        args = ["setprop", prop, value]
        return self.shell(*args, **kwargs).rstrip()

    @timeoutable
    def start(self, package_name: str, activity_name: str = None, timeout: TimeoutType = None, **kwargs) -> str:
        """
        启动app的launcher页面
        :param package_name: 包名
        :param activity_name: activity名
        :return: adb输出结果
        """
        if not activity_name:
            app = self.get_app(package_name, detail=True, timeout=timeout)
            activity = app.get_launch_activity()
            if not activity:
                raise AdbError(f"App {app.name} does not have a launch activity")
            activity_name = activity.name

        return self.shell(
            "am", "start",
            "-a", "android.intent.action.MAIN",
            "-c", "android.intent.category.LAUNCHER",
            "-n", f"{package_name}/{activity_name}",
            **kwargs
        )

    @timeoutable
    def kill(self, package_name: str, **kwargs) -> str:
        """
        关闭进程
        :param package_name: 关闭的包名
        :return: adb输出结果
        """
        args = ["am", "kill", package_name]
        return self.shell(*args, **kwargs).rstrip()

    @timeoutable
    def force_stop(self, package_name: str, **kwargs) -> str:
        """
        关闭进程
        :param package_name: 关闭的包名
        :return: adb输出结果
        """
        args = ["am", "force-stop", package_name]
        return self.shell(*args, **kwargs).rstrip()

    @timeoutable
    def is_file_exist(self, path: str, **kwargs) -> bool:
        """
        文件是否存在
        :param path: 文件路径
        :return: 是否存在
        """
        args = ["[", "-a", path, "]", "&&", "echo", "-n", "1"]
        out = self.shell(*args, **kwargs)
        return utils.bool(utils.int(out, default=0), default=False)

    @timeoutable
    def is_directory_exist(self, path: str, **kwargs) -> bool:
        """
        文件夹是否存在
        :param path: 文件夹路径
        :return: 是否存在
        """
        args = ["[", "-d", path, "]", "&&", "echo", "-n", "1"]
        out = self.shell(*args, **kwargs)
        return utils.bool(utils.int(out, default=0), default=False)

    @cached_classproperty
    def _agent_info(self) -> dict:
        agent_path = environ.get_asset_path("android-tools.json")
        agent_data = json.loads(utils.read_file(agent_path, text=True))
        return agent_data["AGENT_APK"]

    @cached_property
    def _agent_path(self) -> str:
        """
        初始化agent
        :return: agent路径
        """
        apk_name = self._agent_info["name"]
        apk_md5 = self._agent_info["md5"]

        apk_path = environ.get_asset_path(apk_name)
        target_dir = self.get_data_path("android", "agent", apk_md5)
        target_path = self.push_file(apk_path, target_dir, apk_name, skip_exist=True)
        if not self.is_file_exist(target_path):
            raise AdbError("%s does not exist" % target_path)

        return target_path

    def make_agent_args(
            self,
            *args: str,
            app_name: str = None,
            app_path: str = None,
            data_path: str = None,
            library_path: str = None,
            plugin_path: str = None,
    ) -> [str]:
        """
        生成agent参数
        :param args: 参数
        :param app_name: 伪造的包名
        :param app_path: 伪造的包路径
        :param data_path: context.getDataDir()路径
        :param library_path: LD_LIBRARY_PATH路径
        :param plugin_path: 插件路径
        :return: 参数列表
        """
        agent_args = list()
        agent_args.append(f"CLASSPATH={self._agent_path}")
        if app_name:
            agent_args.append(f"APP_PACKAGE_NAME={app_name}")
        if app_path:
            agent_args.append(f"APP_PATH={app_path}")
        if data_path:
            agent_args.append(f"APP_DATA_PATH={data_path}")
        if library_path:
            agent_args.append(f"APP_LIBRARY_PATH={library_path}")
            agent_args.append(f"LD_LIBRARY_PATH={library_path}:$LD_LIBRARY_PATH")
        if plugin_path:
            agent_args.append(f"AGENT_PLUGIN_PATH={plugin_path}")
        agent_args.extend(["app_process", "/", self._agent_info["main"]])
        agent_args.extend(args)

        return agent_args

    @timeoutable
    def call_agent(
            self,
            *args: str,
            app_name: str = None,
            app_path: str = None,
            data_path: str = None,
            library_path: str = None,
            plugin_path: str = None,
            **kwargs
    ) -> str:
        """
        调用辅助agent功能
        :param args: 参数
        :param app_name: 伪造的包名
        :param app_path: 伪造的包路径
        :param data_path: context.getDataDir()路径
        :param library_path: LD_LIBRARY_PATH路径
        :param plugin_path: 插件路径
        :return: 输出结果
        """
        # call apk
        result = self.shell(
            *self.make_agent_args(
                *args,
                app_name=app_name,
                app_path=app_path,
                data_path=data_path,
                library_path=library_path,
                plugin_path=plugin_path,
            ),
            **kwargs
        )

        if result:
            match = _agent_output_pattern.search(result)
            if match is None:
                raise AdbError(result)
            index = match.span()[1]
            result = result[index:]

        return result

    @timeoutable
    def get_current_package(self, **kwargs) -> str:
        """
        获取顶层包名
        :return: 顶层包名
        """
        if self.uid < 10000:
            args = ["dumpsys", "activity", "top", "|", "grep", "^TASK", "-A", "1", ]
            out = self.shell(*args, **kwargs)
            items = out.splitlines()[-1].split()
            if items is not None and len(items) >= 2:
                return items[1].split("/")[0].rstrip()
        # use agent instead of dumpsys
        out = self.call_agent("common", "--top-package", **kwargs).strip()
        if not utils.is_empty(out):
            return out
        raise AdbError("can not fetch top package")

    @timeoutable
    def get_current_activity(self, **kwargs) -> str:
        """
        获取顶层activity名
        :return: 顶层activity名
        """
        args = ["dumpsys", "activity", "top", "|", "grep", "^TASK", "-A", "1"]
        result = self.shell(*args, **kwargs)
        items = result.splitlines()[-1].split()
        if items is not None and len(items) >= 2:
            return items[1].rstrip()
        raise AdbError("can not fetch top activity")

    @timeoutable
    def get_apk_path(self, package_name: str, **kwargs) -> str:
        """
        获取apk路径
        :return: apk路径
        """
        if self.uid < 10000:
            out = self.shell("pm", "path", package_name, **kwargs)
            match = re.search(r"^.*package:\s*(.*)[\s\S]*$", out)
            if match is not None:
                return match.group(1).strip()
        obj = self.get_apps(package_name, **kwargs)
        return utils.get_item(obj, 0, "sourceDir", default="")

    @timeoutable
    def get_uid(self, package_name: str = None, timeout: TimeoutType = None) -> int:
        """
        根据包名获取uid
        :param package_name: 包名，为空则返回当前uid
        :param timeout: 超时时间
        :return: uid
        """
        if package_name:
            app = self.get_app(package_name, timeout=timeout)
            return app.user_id
        else:
            out = self.shell("id", "-u", timeout=timeout)
            uid = utils.int(out.strip(), default=-1)
            if uid >= 0:
                return uid
            out = self.shell("echo", "-n", "${USER_ID}", timeout=timeout)
            uid = utils.int(out.strip(), default=-1)
            if uid >= 0:
                return uid
            raise AdbError("unknown adb uid: %s" % out)

    @timeoutable
    def get_app(self, package_name: str, detail: bool = None, **kwargs) -> App:
        """
        根据包名获取包信息
        :param package_name: 包名
        :param detail: 获取详细信息
        :return: 包信息
        """
        args = ["package", "--packages", package_name]
        if detail is True:
            args.append("--detail")
        objs = json.loads(self.call_agent(*args, **kwargs))
        if len(objs) == 0:
            raise AdbError(f"App '{package_name}' not found")
        return App(objs[0])

    @timeoutable
    def get_apps(self, *package_names: str, system: bool = None, detail: bool = False, **kwargs) -> [App]:
        """
        获取包信息
        :param package_names: 需要匹配的所有包名，为空则匹配所有
        :param system: true只匹配系统应用，false只匹配非系统应用，为空则全匹配
        :param detail: 获取详细信息
        :return: 包信息
        """
        result = []
        agent_args = ["package"]
        if not utils.is_empty(package_names):
            agent_args.append("--packages")
            agent_args.extend(package_names)
        if system is True:
            agent_args.append("--system")
        elif system is False:
            agent_args.append("--non-system")
        if detail is True:
            agent_args.append("--detail")
        objs = json.loads(self.call_agent(*agent_args, **kwargs))
        for obj in objs:
            result.append(App(obj))
        return result

    @timeoutable
    def get_apps_for_uid(self, *uids: int, detail: bool = False, **kwargs) -> [App]:
        """
        获取指定uid包信息
        :param uids: 需要匹配的所有uid
        :param detail: 获取详细信息
        :return: 包信息
        """
        result = []
        agent_args = ["package"]
        if not utils.is_empty(uids):
            agent_args.append("--uids")
            agent_args.extend([str(uid) for uid in uids])
        if detail is True:
            agent_args.append("--detail")
        objs = json.loads(self.call_agent(*agent_args, **kwargs))
        for obj in objs:
            result.append(App(obj))
        return result

    @timeoutable
    def get_system_service(self, service_name: str, detail: bool = None, **kwargs) -> SystemService:
        """
        根据服务名获取系统服务信息
        :param service_name: 服务名
        :param detail: 获取详细信息
        :return: 包信息
        """
        args = ["service", "--names", service_name]
        if detail is True:
            args.append("--detail")
        objs = json.loads(self.call_agent(*args, **kwargs))
        if len(objs) == 0:
            raise AdbError(f"Service '{service_name}' not found")
        return SystemService(objs[0])

    @timeoutable
    def get_system_services(self, *service_names: str, detail: bool = False, **kwargs) -> [SystemService]:
        """
        获取系统服务信息
        :param service_names: 服务名（不填则全量）
        :param detail: 获取详细信息
        :return: 包信息
        """
        result = []
        agent_args = ["service"]
        if service_names:
            agent_args.append("--names")
            agent_args.extend(service_names)
        if detail is True:
            agent_args.append("--detail")
        objs = json.loads(self.call_agent(*agent_args, **kwargs))
        for obj in objs:
            result.append(SystemService(obj))
        return result

    @timeoutable
    def list_tcp_sockets(self, **kwargs) -> [InetSocket]:
        """
        同netstat命令，获取设备tcp连接情况，需要读取/proc/net/tcp文件，高版本设备至少需要shell权限
        :return: tcp连接列表
        """
        return self._list_sockets(InetSocket, ["common", "--list-tcp-sock"], **kwargs)

    @timeoutable
    def list_udp_sockets(self, **kwargs) -> [InetSocket]:
        """
        同netstat命令，获取设备udp连接情况，需要读取/proc/net/udp文件，高版本设备至少需要shell权限
        :return: udp连接列表
        """
        return self._list_sockets(InetSocket, ["common", "--list-udp-sock"], **kwargs)

    @timeoutable
    def list_raw_sockets(self, **kwargs) -> [InetSocket]:
        """
        同netstat命令，获取设备raw连接情况，需要读取/proc/net/raw文件，高版本设备至少需要shell权限
        :return: raw连接列表
        """
        return self._list_sockets(InetSocket, ["common", "--list-raw-sock"], **kwargs)

    @timeoutable
    def list_unix_sockets(self, **kwargs) -> [UnixSocket]:
        """
        同netstat命令，获取设备unix连接情况，需要读取/proc/net/unix文件，高版本设备至少需要shell权限
        :return: unix连接列表
        """
        return self._list_sockets(UnixSocket, ["common", "--list-unix-sock"], **kwargs)

    @timeoutable
    def _list_sockets(self, type, args, **kwargs):
        result = []
        objs = json.loads(self.call_agent(*args, **kwargs))
        for obj in objs:
            result.append(type(obj))
        return result

    @timeoutable
    def list_processes(self, **kwargs) -> [Process]:
        """
        列出所有进程
        """
        result = []
        agent_args = ["common", "--list-process"]
        objs = json.loads(self.call_agent(*agent_args, **kwargs))
        for obj in objs:
            result.append(Process(obj))
        return result

    @timeoutable
    def list_files(self, path: str, **kwargs) -> [File]:
        """
        列出指定目录下的所有文件
        """
        result = []
        agent_args = ["common", "--list-file", path]
        objs = json.loads(self.call_agent(*agent_args, **kwargs))
        for obj in objs:
            result.append(File(obj))
        return result

    @timeoutable
    def get_clipboard(self, **kwargs) -> str:
        """
        获取剪切板内容
        """
        return self.call_agent("common", "--get-clipboard", **kwargs)

    @timeoutable
    def set_clipboard(self, text, **kwargs) -> str:
        """
        设置剪切板内容
        """
        return self.call_agent("common", "--set-clipboard", text, **kwargs)

    @cached_property
    def _data_path(self):
        data_path = f"/data/local/tmp/{environ.name}_{self.uid}"
        if self.uid < 10000:
            return data_path
        adb_data_path = self.shell("echo", "-n", "$ADB_DATA_PATH").strip()
        if not adb_data_path:
            return f"/data/local/tmp/{environ.name}"
        return data_path

    def get_data_path(self, *names: [str]) -> str:
        """
        /data/local/tmp 路径
        :param names: 文件名
        :return: 路径
        """
        return self.join_path(self._data_path, *names)

    @classmethod
    def join_path(cls, *names: str):
        path = ""
        for name in names:
            path += "/" + cls.get_safe_path(name).strip("/")
        return path

    @classmethod
    def get_base_name(cls, path: str) -> str:
        index = path.rfind("/")
        if index >= 0:
            return path[index + 1:]
        return path

    @classmethod
    def get_safe_path(cls, path: str) -> str:
        """
        过滤"../"关键字
        :param path: 原始路径
        :return: 过滤完"../"的路径
        """
        temp = path
        while True:
            result = temp.replace("../", "..")
            if temp == result:
                return result
            temp = result

    def __repr__(self):
        return f"AdbDevice<{self.id}>"


class AdbForward(Stoppable):
    local = property(fget=lambda self: self._local)
    local_port = property(fget=lambda self: int(self._local[1]))
    remote = property(fget=lambda self: self._remote)
    remote_port = property(fget=lambda self: int(self._remote[1]))

    def __init__(self, device: AdbDevice, local: str, remote: str):
        self._device = device
        self._local = None
        self._remote = None

        def start():
            nonlocal local
            result = self._device.exec("forward", local, remote)
            if local == "tcp:0":
                local = f"tcp:{result}"
            self._local = local.split(":", maxsplit=1)
            self._remote = remote.split(":", maxsplit=1)

        self._stop_on_error(start)

    def stop(self):
        if self._local is not None:
            self._device.exec("forward", "--remove", ":".join(self._local), ignore_errors=True)


class AdbReverse(Stoppable):
    local = property(fget=lambda self: self._local)
    local_port = property(fget=lambda self: int(self._local[1]))
    remote = property(fget=lambda self: self._remote)
    remote_port = property(fget=lambda self: int(self._remote[1]))

    def __init__(self, device: AdbDevice, remote: str, local: str):
        self._device = device
        self._local = None
        self._remote = None

        def start():
            nonlocal remote
            result = self._device.exec("reverse", remote, local)
            if remote == "tcp:0":
                remote = f"tcp:{result}"
            self._local = local.split(":", maxsplit=1)
            self._remote = remote.split(":", maxsplit=1)
            return self

        self._stop_on_error(start)

    def stop(self):
        if self._remote is not None:
            self._device.exec("reverse", "--remove", ":".join(self._remote), ignore_errors=True)


class AdbRedirect(Stoppable):
    address = property(fget=lambda self: self._address)
    port = property(fget=lambda self: self._port)

    def __init__(self, device: AdbDevice, address: str, port: int, uid: int):
        self._device = device
        self._address = None
        self._port = None
        self._reverse_port = None

        def start():
            self._port = port
            if not address:
                # 如果没有指定目标地址，则通过reverse端口访问
                self._reverse_port = self._device.exec("reverse", f"tcp:0", f"tcp:{port}").strip()
                self._address = "127.0.0.1"
                destination = f"{self._address}:{self._reverse_port}"
            else:
                # 指定了目标地址那就直接用目标地址
                self._address = address
                destination = f"{address}:{port}"

            _logger.debug(f"Redirect all traffic to {destination}")

            # 配置iptables规则，首先清除之前的规则，然后再把流量转发到目标端口上
            self._device.sudo("iptables", "-t", "nat", "-F")

            args = ["-A", "OUTPUT", "-p", "tcp"]  # 添加一条tcp协议的转发规则
            args += ["!", "-o", "lo"]  # 过滤localhost
            if uid is not None:
                args += ["-m", "owner", "--uid-owner", uid]  # 指定要重定向流量的uid
            args += ["-j", "DNAT", "--to-destination", destination]  # 转发到指定端口
            self._device.sudo("iptables", "-t", "nat", *args)

        self._stop_on_error(start)

    def stop(self):
        # 清空iptables -t nat配置
        self._device.sudo("iptables", "-t", "nat", "-F", ignore_errors=True)
        # 如果占用reverse端口，则释放端口
        if self._reverse_port:
            self._device.exec("reverse", "--remove", f"tcp:{self._reverse_port}", ignore_errors=True)
