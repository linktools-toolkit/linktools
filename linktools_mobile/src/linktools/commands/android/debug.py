#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : at_debug.py
@time    : 2019/04/22
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

from linktools import utils
from linktools.cli import CommandParser
from linktools.mobile.cli import AndroidCommand, AndroidNamespace


class Command(AndroidCommand):
    """
    Debug Android apps effectively using the Java Debugger (jdb)
    """

    @property
    def _description(self) -> str:
        return "debugger"

    def init_arguments(self, parser: CommandParser) -> None:
        parser.add_argument('package', action='store', default=None,
                            help='regular expression')
        parser.add_argument('activity', action='store', default=None,
                            help='regular expression')
        parser.add_argument('-p', '--port', action='store', type=int, default=8701,
                            help='fetch all apps')

    def run(self, args: AndroidNamespace) -> Optional[int]:
        device = args.device_selector.select()

        device.shell("am", "force-stop", args.package, log_output=True)
        device.shell("am", "start", "-D", "-n", "{}/{}".format(args.package, args.activity), log_output=True)

        pid = utils.int(device.shell("top", "-n", "1", "|", "grep", args.package).split()[0])
        with device.forward(f"tcp:{args.port}", f"jdwp:{pid}"):
            data = input("jdb connect? [Y/n]: ").strip()
            if data in ["", "Y", "y"]:
                process = utils.popen(
                    "jdb", "-connect", f"com.sun.jdi.SocketAttach:hostname=127.0.0.1,port={args.port}")
                return process.call()


command = Command()
if __name__ == "__main__":
    command.main()
