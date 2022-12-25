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
   /  oooooooooooooooo  .o.  oooo /,   \,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,`\--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""
from argparse import ArgumentParser
from typing import Optional

from linktools import utils


class Script(utils.AndroidScript):

    @property
    def _description(self) -> str:
        return "debugger"

    def _add_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument('package', action='store', default=None,
                            help='regular expression')
        parser.add_argument('activity', action='store', default=None,
                            help='regular expression')
        parser.add_argument('-p', '--port', action='store', type=int, default=8701,
                            help='fetch all apps')

    def _run(self, args: [str]) -> Optional[int]:
        args = self.argument_parser.parse_args(args)
        device = args.parse_device()

        device.shell("am", "force-stop", args.package, output_to_logger=True)
        device.shell("am", "start", "-D", "-n", "{}/{}".format(args.package, args.activity), output_to_logger=True)

        pid = utils.int(device.shell("top", "-n", "1", "|", "grep", args.package).split()[0])
        with device.forward(f"tcp:{args.port}", f"jdwp:{pid}"):
            data = input("jdb connect? [Y/n]: ").strip()
            if data in ["", "Y", "y"]:
                process = utils.Popen("jdb", "-connect",
                                      "com.sun.jdi.SocketAttach:hostname=127.0.0.1,port={}".format(args.port))
                return process.call()


script = Script()
if __name__ == '__main__':
    script.main()
