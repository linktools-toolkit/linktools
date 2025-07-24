#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : at_adb.py
@time    : 2019/03/04
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
from typing import Optional

from linktools.cli import CommandParser, CommandMain, AndroidCommand, AndroidNamespace


class Command(AndroidCommand):
    """
    Manage multiple Android devices effortlessly with adb commands
    """

    _GENERAL_COMMANDS = [
        "devices",
        "help",
        "version",
        "connect",
        "disconnect",
        "keygen",
        # "wait-for-",
        "start-server",
        "kill-server",
        "reconnect",
        "attach",
        "detach",
    ]

    @property
    def main(self) -> CommandMain:
        return CommandMain(self, show_log_time=False, show_log_level=False)

    def init_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("adb_args", nargs="...", metavar="args", help="adb args")

    def run(self, args: AndroidNamespace) -> Optional[int]:
        adb_args = args.adb_args
        if adb_args and adb_args[0] not in self._GENERAL_COMMANDS and not adb_args[0].startswith("wait-for-"):
            device = args.device_selector.select()
            process = device.popen(*adb_args, capture_output=False)
            return process.call()

        adb = args.device_selector.bridge
        process = adb.popen(*adb_args, capture_output=False)
        return process.call()


command = Command()
if __name__ == "__main__":
    command.main()
