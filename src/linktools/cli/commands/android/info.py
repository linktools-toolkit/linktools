#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : info.py 
@time    : 2022/07/31
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
from argparse import Namespace
from typing import Optional

from linktools import environ
from linktools.cli import AndroidCommand, CommandParser

props = (
    "ro.product.manufacturer",
    "ro.product.model",
    "ro.product.name",
    "ro.board.platform",
    "ro.build.version.release",
    "ro.build.version.base_os",
    "ro.build.version.sdk",
    "ro.build.version.incremental",
    "ro.build.version.security_patch",
    "ro.odm.build.id",
    "ro.build.fingerprint",
    "ro.build.build.fingerprint",
    "ro.bootimage.build.fingerprint",
    "ro.odm.build.fingerprint",
    "ro.product.build.fingerprint",
    "ro.system_ext.build.fingerprint",
    "ro.system.build.fingerprint",
    "ro.vendor.build.fingerprint",
)

files = (
    "/proc/sys/kernel/random/boot_id",  # 重启后变化
    "/proc/sys/kernel/random/uuid",
    "/sys/block/mmcblk0/device/cid",
    "/sys/devices/soc0/serial_number",
    "/proc/misc",
    "/proc/version",
)

cmds = (
    ("uname", ("uname -a",)),
    ("magisk df", ("df | grep /sbin/.magisk",)),
    ("magisk mount", ("mount | grep /sbin/.magisk",)),
    ("magisk process", ("df | ps | grep magisk",)),
    ("ip", ("ip a",)),
)


class Command(AndroidCommand):
    """
    Collect detailed device information
    """

    def init_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('agent_args', nargs='...', help="agent args")

    def run(self, args: Namespace) -> Optional[int]:
        device = args.device_picker.pick()

        environ.logger.info(f"Property", style="red")
        for prop in props:
            environ.logger.info(
                f"{prop}: {device.get_prop(prop)}",
                indent=2
            )

        environ.logger.info(f"File", style="red")
        for file in files:
            environ.logger.info(
                f"{file}: {device.shell('cat', file, ignore_errors=True).strip()}",
                indent=2
            )

        environ.logger.info(f"Cmdline", style="red")
        for cmd in cmds:
            environ.logger.info(
                f"{cmd[0]}: {device.shell(*cmd[1], ignore_errors=True).strip()}",
                indent=2
            )

        return


command = Command()
if __name__ == "__main__":
    command.main()
