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
   /  oooooooooooooooo  .o.  oooo /,   \,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,`\--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""
from argparse import ArgumentParser
from typing import Optional

from linktools import utils
from linktools.android import Adb


class Script(utils.AndroidScript):

    GENERAL_COMMANDS = [
        "devices",
        "help",
        "version",
        "connect",
        "disconnect",
        "keygen",
        "wait-for-",
        "start-server",
        "kill-server",
        "reconnect",
    ]

    @property
    def _description(self) -> str:
        return "adb wrapper"

    def _add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('adb_args', nargs='...', help="adb args")

    def _run(self, args: [str]) -> Optional[int]:
        args, extra = self.argument_parser.parse_known_args(args)

        adb_args = [*extra, *args.adb_args]
        if not extra:
            if args.adb_args and args.adb_args[0] not in self.GENERAL_COMMANDS:
                device = args.parse_device()
                process = device.popen(*adb_args, capture_output=False)
                return process.call()

        process = Adb.popen(*adb_args, capture_output=False)
        return process.call()


script = Script()
if __name__ == '__main__':
    script.main()
