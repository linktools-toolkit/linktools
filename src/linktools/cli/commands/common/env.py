#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : container.py
@time    : 2024/3/21
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
from typing import Any

from linktools.cli import subcommand, SubCommandWrapper, BaseCommandGroup
from linktools.cli.env import get_commands


class InitCommand(BaseCommandGroup):
    """
    initialize environment
    """

    @property
    def name(self) -> str:
        return "init"

    @subcommand("android", help="initialize android environment")
    def on_init_android(self):
        try:
            self.logger.info("initialize adb ...")
            self.environ.tools["adb"].prepare()
        except Exception as e:
            self.logger.warning(f"initialize adb failed: {e}")

        try:
            from linktools.frida import FridaAndroidServer
            self.logger.info("initialize android frida server ...")
            FridaAndroidServer.setup(abis=["arm", "arm64"])
        except Exception as e:
            self.logger.warning(f"initialize android frida server failed: {e}")

    @subcommand("ios", help="initialize ios environment")
    def on_init_ios(self):
        try:
            self.logger.info("initialize go-ios ...")
            self.environ.tools["ios"].prepare()
        except Exception as e:
            self.logger.warning(f"initialize go-ios failed: {e}")


class Command(BaseCommandGroup):
    """
    Manage and configure the Linktools environment
    """

    def init_subcommands(self) -> Any:
        return [
            SubCommandWrapper(InitCommand()),
            get_commands(self.environ),
        ]


command = Command()
if __name__ == "__main__":
    command.main()
