#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : ssh.py 
@time    : 2023/12/17
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
import time
from argparse import Namespace
from typing import Optional

from linktools import utils
from linktools.cli import subcommand, subcommand_argument, IOSCommand, CommandParser
from linktools.mobile.ios import GoIOSDevice


class Command(IOSCommand):

    @subcommand("reverse", pass_args=True)
    @subcommand_argument("--local-port")
    @subcommand_argument("--remote-port")
    def on_reverse(self, args: Namespace, local_port: int = 8000, remote_port: int = None):
        device: GoIOSDevice = args.device_picker.pick()
        with device.ssh() as client:
            with client.reverse(forward_host="localhost", forward_port=local_port, remote_port=remote_port) as reverse:
                self.logger.info(f"Reverse port: {reverse.remote_port}")
                time.sleep(1000)

    @subcommand("forward", pass_args=True)
    @subcommand_argument("--local-port")
    @subcommand_argument("--remote-port")
    def on_forward(self, args: Namespace, local_port: int = None, remote_port: int = 22):
        device: GoIOSDevice = args.device_picker.pick()
        with device.forward(local_port=local_port or utils.get_free_port(), remote_port=remote_port) as forward:
            self.logger.info(f"Forward port: {forward.local_port}")
            time.sleep(1000)

    def init_arguments(self, parser: CommandParser) -> None:
        self.add_subcommands(parser)

    def run(self, args: Namespace) -> Optional[int]:
        return self.run_subcommand(args)


command = Command()
if __name__ == '__main__':
    command.main()
