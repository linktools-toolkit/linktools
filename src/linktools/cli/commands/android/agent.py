#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : at_call_agent.py
@time    : 2018/12/02
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
import os
from argparse import ArgumentParser, Namespace
from typing import Optional

from linktools.android import AdbDevice
from linktools.cli import AndroidCommand, CommandError


class AgentDevice(AdbDevice):

    def get_agent_path(self, *name: str) -> str:
        return self.get_data_path("android", "agent", *name)

    def push_agent_plugin(self, src_path: str = None) -> Optional[str]:
        if not src_path:
            return None
        if not os.path.exists(src_path):
            raise CommandError(f"Plugin file not found: {src_path}")
        return self.push_file(src_path, self.get_agent_path("plugin"))


class Command(AndroidCommand):
    """
    Debug and interact with android-tools.apk for troubleshooting
    """

    def init_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("-p", "--privilege", action="store_true", default=False,
                            help="run with root privilege")
        parser.add_argument("-u", "--user", action="store",
                            help="run with user privilege")
        parser.add_argument("--data", metavar="PATH", action="store",
                            help="data path")
        parser.add_argument("--library", metavar="PATH", action="store",
                            help="library path")
        parser.add_argument("--plugin", metavar="PATH", action="store",
                            help="plugin file path")
        parser.add_argument("agent_args", nargs="...", help="agent args")

    def run(self, args: Namespace) -> Optional[int]:
        device = args.device_picker.pick().copy(AgentDevice)

        agent_args = device.make_agent_args(
            *args.agent_args,
            data_path=args.data or device.get_agent_path("data"),
            library_path=args.library or device.get_agent_path("data", "lib"),
            plugin_path=device.push_agent_plugin(args.plugin),
        )

        shell_args = device.make_shell_args(
            *agent_args,
            privilege=args.privilege,
            user=args.user
        )

        process = device.popen(*shell_args)
        return process.call()


command = Command()
if __name__ == "__main__":
    command.main()
