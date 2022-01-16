#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : argparser.py 
@time    : 2019/03/09
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
import argparse
import functools
import os

from tidevice import Usbmux, Device, MuxError

from linktools import utils, resource, logger
from linktools.argparser import ArgumentParser

_DEVICE_CACHE_PATH = resource.get_data_path("ios_udid_cache.txt", create_parent=True)


class IOSArgumentParser(ArgumentParser):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        class _Context:

            def __init__(self):
                self._usbmux: Usbmux = None

            @property
            def usbmux(self):
                return self._usbmux or Usbmux()

            @usbmux.setter
            def usbmux(self, value: Usbmux):
                self._usbmux = value

        def parse_handler(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                udid = fn(*args, **kwargs)
                if udid is not None:
                    with open(_DEVICE_CACHE_PATH, "wt+") as fd:
                        fd.write(udid)
                return Device(udid, context.usbmux)

            return wrapper

        @parse_handler
        def parse_device():
            usbmux = context.usbmux
            devices = usbmux.device_list()
            if len(devices) == 0:
                raise MuxError("error: no devices/emulators found")

            if len(devices) == 1:
                return devices[0].udid

            logger.message("more than one device/emulator")
            for i in range(len(devices)):
                try:
                    name = Device(devices[0].udid, usbmux).name
                except Exception:
                    name = ""
                logger.message(f"%d: %-20s [%s]" % (i + 1, devices[i].udid, name))
            while True:
                offset = 1
                data = input("enter device index %d~%d (default 1): " % (1, len(devices)))
                if utils.is_empty(data):
                    index = 1 - offset
                    break
                index = utils.cast(int, data, -1) - offset
                if 0 <= index < len(devices):
                    break
            if 0 <= index < len(devices):
                return devices[index].udid

        class UdidAction(argparse.Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @parse_handler
                def wrapper():
                    return str(values)

                setattr(namespace, self.dest, wrapper)

        class IndexAction(argparse.Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @parse_handler
                def wrapper():
                    index = int(values)
                    usbmux = context.usbmux
                    devices = usbmux.device_list()
                    if utils.is_empty(devices):
                        raise MuxError("error: no devices/emulators found")
                    if not 0 < index <= len(devices):
                        raise MuxError("error: index %d out of range %d~%d" % (index, 1, len(devices)))
                    index = index - 1
                    return devices[index].udid

                setattr(namespace, self.dest, wrapper)

        class LastAction(argparse.Action):

            def __call__(self, parser, namespace, values, option_string=None):
                @parse_handler
                def wrapper():
                    if os.path.exists(_DEVICE_CACHE_PATH):
                        with open(_DEVICE_CACHE_PATH, "rt") as fd:
                            result = fd.read().strip()
                            if len(result) > 0:
                                return result
                    raise MuxError("error: no device used last time")

                setattr(namespace, self.dest, wrapper)

        class UsbmuxdAction(argparse.Action):

            def __call__(self, parser, namespace, values, option_string=None):
                context.usbmux = Usbmux(str(values))

        context = _Context()

        group = self.add_argument_group(title="tidevice optional arguments")
        exclusive_group = group.add_mutually_exclusive_group()
        exclusive_group.add_argument("-u", "--udid", metavar="UDID", dest="parse_device", action=UdidAction,
                                     help="specify unique device identifier", default=parse_device)
        exclusive_group.add_argument("-i", "--index", metavar="INDEX", dest="parse_device", action=IndexAction,
                                     help="use device with given index")
        exclusive_group.add_argument("-l", "--last", dest="parse_device", nargs=0, const=True, action=LastAction,
                                     help="use last device")
        group.add_argument("--socket", metavar="SOCKET", dest="parse_usbmux", action=UsbmuxdAction,
                           help="usbmuxd listen address, host:port or local-path")