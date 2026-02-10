#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : device.py 
@time    : 2023/11/12
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

import abc
import functools
from argparse import Action, Namespace
from typing import Optional, Callable, List, Type, Generic

from linktools.cli import BaseCommand, CommandParser
from linktools.rich import choose
from linktools.types import PathType, FileCache
from . import Bridge, BridgeError, BaseDevice, BridgeType, DeviceType, list_devices
from .android import Adb, AdbError, AdbDevice
from .ios import GoIOS, GoIOSError, GoIOSDevice


class DeviceCache:

    def __init__(self, path: PathType, key: str):
        self._cache = FileCache(path)
        self._key = key

    def read(self) -> Optional[str]:
        return self._cache.get(self._key, None)

    def write(self, cache: str) -> None:
        self._cache.set(self._key, cache)

    def __call__(self, fn: Callable[..., BaseDevice]):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            device: BaseDevice = fn(*args, **kwargs)
            if device is not None:
                self.write(device.id)
            return device

        return wrapper


class DeviceSelector(Generic[BridgeType, DeviceType]):

    def __init__(self, func: Callable[[BridgeType], DeviceType] = None, options: List[str] = None):
        self.func = func
        self.options = options or []
        self._const = True

    @property
    def bridge(self) -> Optional[BridgeType]:
        return None

    def select(self) -> DeviceType:
        return self.func(self.bridge)

    def __call__(self) -> DeviceType:
        return self.func(self.bridge)

    @classmethod
    def copy_on_write(cls, namespace: Namespace, dest: str) -> "DeviceSelector":
        if hasattr(namespace, dest):
            selector: DeviceSelector = getattr(namespace, dest)
            if not selector:
                selector = cls()
                selector._const = False
                setattr(namespace, dest, selector)
            elif selector._const:
                new_parser = cls()
                new_parser.func = selector.func
                new_parser.options = list(selector.options)
                new_parser._const = False
                setattr(namespace, dest, new_parser)
                selector = new_parser
        else:
            selector = cls()
            selector._const = False
            setattr(namespace, dest, selector)
        return selector


class AndroidSelector(DeviceSelector[Adb, AdbDevice]):

    @property
    def bridge(self) -> Adb:
        return Adb(self.options)


class IOSSelector(DeviceSelector[GoIOS, GoIOSDevice]):

    @property
    def bridge(self) -> GoIOS:
        return GoIOS(self.options)


class DeviceCommandMixin:

    def add_device_options(self: "BaseCommand", parser: CommandParser):

        parser = parser or self._argument_parser
        prefix = parser.prefix_chars[0] if parser.prefix_chars else "-"
        cache = DeviceCache(self.environ.get_temp_path("commands", "cache"), "device")

        @cache
        def select(bridge: Bridge):
            devices = tuple(list_devices(alive=True))
            if len(devices) == 0:
                raise BridgeError("no devices/emulators found")

            if len(devices) == 1:
                return devices[0]

            return choose(
                "Choose device",
                title="More than one device/emulator",
                choices={device: device.pretty_id for device in devices},
                default=devices[0]
            )

        class IDAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @cache
                def select(bridge: Bridge):
                    device_id = str(values)
                    for device in list_devices():
                        if device.id == device_id:
                            return device
                    raise BridgeError(f"no devices/emulators with {device_id} found")

                device_parser = DeviceSelector.copy_on_write(namespace, self.dest)
                device_parser.func = select

        class LastAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @cache
                def select(bridge: Bridge):
                    device_id = cache.read()
                    if not device_id:
                        raise BridgeError("no device used last time")
                    for device in list_devices():
                        if device.id == device_id:
                            return device
                    raise BridgeError("no device used last time")

                device_parser = DeviceSelector.copy_on_write(namespace, self.dest)
                device_parser.func = select

        option_group = parser.add_argument_group(title="mobile device options")
        option_group.set_defaults(device_selector=DeviceSelector(select))

        device_group = option_group.add_mutually_exclusive_group()
        device_group.add_argument(f"{prefix}i", f"{prefix}{prefix}id", metavar="ID", dest="device_selector",
                                  action=IDAction, help="specify unique device identifier")
        device_group.add_argument(f"{prefix}l", f"{prefix}{prefix}last", dest="device_selector", nargs=0, const=True,
                                  action=LastAction, help="use last device")


class AndroidCommandMixin:

    def add_android_options(self: BaseCommand, parser: CommandParser) -> None:

        parser = parser or self._argument_parser
        prefix = parser.prefix_chars[0] if parser.prefix_chars else "-"
        cache = DeviceCache(self.environ.get_temp_path("commands", "cache"), "android")

        @cache
        def select(adb: Adb):
            devices = tuple(adb.list_devices(alive=True))
            if len(devices) == 0:
                raise AdbError("no devices/emulators found")

            if len(devices) == 1:
                return devices[0]

            return choose(
                "Choose device",
                title="More than one device/emulator",
                choices={device: device.pretty_id for device in devices},
                default=devices[0]
            )

        class SerialAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @cache
                def select(adb: Adb):
                    return AdbDevice(str(values), adb=adb)

                device_parser = AndroidSelector.copy_on_write(namespace, self.dest)
                device_parser.func = select

        class DeviceAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @cache
                def select(adb: Adb):
                    return AdbDevice(adb.exec("-d", "get-serialno").strip(" \r\n"), adb=adb)

                device_parser = AndroidSelector.copy_on_write(namespace, self.dest)
                device_parser.func = select

        class EmulatorAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @cache
                def select(adb: Adb):
                    return AdbDevice(adb.exec("-e", "get-serialno").strip(" \r\n"), adb=adb)

                device_parser = AndroidSelector.copy_on_write(namespace, self.dest)
                device_parser.func = select

        class ConnectAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @cache
                def select(adb: Adb):
                    addr = str(values)
                    if addr.find(":") < 0:
                        addr = addr + ":5555"
                    if addr not in [device.id for device in adb.list_devices()]:
                        adb.exec("connect", addr, log_output=True)
                    return AdbDevice(addr, adb=adb)

                device_parser = AndroidSelector.copy_on_write(namespace, self.dest)
                device_parser.func = select

        class LastAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @cache
                def select(adb: Adb):
                    device_id = cache.read()
                    if device_id:
                        return AdbDevice(device_id, adb=adb)
                    raise AdbError("no device used last time")

                device_parser = AndroidSelector.copy_on_write(namespace, self.dest)
                device_parser.func = select

        class OptionAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                device_parser = AndroidSelector.copy_on_write(namespace, self.dest)
                device_parser.options.append(option_string)
                if isinstance(values, str):
                    device_parser.options.append(values)
                elif isinstance(values, (list, tuple, set)):
                    device_parser.options.extend(values)
                else:
                    device_parser.options.append(str(values))

        option_group = parser.add_argument_group(title="adb options")
        option_group.set_defaults(device_selector=AndroidSelector(select))

        option_group.add_argument(f"{prefix}a", f"{prefix}{prefix}all-interfaces", dest="device_selector", nargs=0,
                                  action=OptionAction,
                                  help="listen on all network interfaces, not just localhost (adb -a option)")

        device_group = option_group.add_mutually_exclusive_group()
        device_group.add_argument(f"{prefix}d", f"{prefix}{prefix}device", dest="device_selector", nargs=0,
                                  action=DeviceAction,
                                  help="use USB device (adb -d option)")
        device_group.add_argument(f"{prefix}s", f"{prefix}{prefix}serial", metavar="SERIAL", dest="device_selector",
                                  action=SerialAction,
                                  help="use device with given serial (adb -s option)")
        device_group.add_argument(f"{prefix}e", f"{prefix}{prefix}emulator", dest="device_selector", nargs=0,
                                  action=EmulatorAction,
                                  help="use TCP/IP device (adb -e option)")
        device_group.add_argument(f"{prefix}c", f"{prefix}{prefix}connect", metavar="IP[:PORT]", dest="device_selector",
                                  action=ConnectAction,
                                  help="use device with TCP/IP")
        device_group.add_argument(f"{prefix}l", f"{prefix}{prefix}last", dest="device_selector", nargs=0,
                                  action=LastAction,
                                  help="use last device")

        option_group.add_argument(f"{prefix}t", f"{prefix}{prefix}transport", metavar="ID", dest="device_selector",
                                  action=OptionAction,
                                  help="use device with given transport ID (adb -t option)")
        option_group.add_argument(f"{prefix}H", metavar="HOST", dest="device_selector", action=OptionAction,
                                  help="name of adb server host [default=localhost] (adb -H option)")
        option_group.add_argument(f"{prefix}P", metavar="PORT", dest="device_selector", action=OptionAction,
                                  help="port of adb server [default=5037] (adb -P option)")
        option_group.add_argument(f"{prefix}L", metavar="SOCKET", dest="device_selector", action=OptionAction,
                                  help="listen on given socket for adb server [default=tcp:localhost:5037] (adb -L option)")


class IOSCommandMixin:

    def add_ios_options(self: BaseCommand, parser: CommandParser):

        parser = parser or self._argument_parser
        prefix = parser.prefix_chars[0] if parser.prefix_chars else "-"
        cache = DeviceCache(self.environ.get_temp_path("commands", "cache"), "ios")

        @cache
        def select(ios: GoIOS):
            devices = tuple(ios.list_devices(alive=True))
            if len(devices) == 0:
                raise GoIOSError("no devices/emulators found")

            if len(devices) == 1:
                return devices[0]

            return choose(
                "Choose device",
                title="More than one device/emulator",
                choices={device: device.pretty_id for device in devices},
                default=devices[0]
            )

        class UdidAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @cache
                def select(ios: GoIOS):
                    return GoIOSDevice(str(values), ios=ios)

                device_parser = IOSSelector.copy_on_write(namespace, self.dest)
                device_parser.func = select

        class LastAction(Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @cache
                def select(ios: GoIOS):
                    device_id = cache.read()
                    if device_id:
                        return GoIOSDevice(device_id, ios=ios)
                    raise GoIOSError("no device used last time")

                device_parser = IOSSelector.copy_on_write(namespace, self.dest)
                device_parser.func = select

        option_group = parser.add_argument_group(title="ios options")
        option_group.set_defaults(device_selector=IOSSelector(select))

        device_group = option_group.add_mutually_exclusive_group()
        device_group.add_argument(f"{prefix}u", f"{prefix}{prefix}udid", metavar="UDID", dest="device_selector",
                                  action=UdidAction,
                                  help="specify unique device identifier")
        device_group.add_argument(f"{prefix}l", f"{prefix}{prefix}last", dest="device_selector", nargs=0, const=True,
                                  action=LastAction,
                                  help="use last device")


class AndroidNamespace(Namespace):
    device_selector: AndroidSelector


class IOSNamespace(Namespace):
    device_selector: IOSSelector


class AndroidCommand(BaseCommand, metaclass=abc.ABCMeta):

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        return super().known_errors + [AdbError]

    def init_base_arguments(self, parser: CommandParser):
        super().init_base_arguments(parser)
        AndroidCommandMixin.add_android_options(self, parser)


class IOSCommand(BaseCommand, metaclass=abc.ABCMeta):

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        return super().known_errors + [GoIOSError]

    def init_base_arguments(self, parser: CommandParser):
        super().init_base_arguments(parser)
        IOSCommandMixin.add_ios_options(self, parser)
