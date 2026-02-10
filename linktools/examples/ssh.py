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

import paramiko

from linktools.cli import BaseCommand, subcommand, subcommand_argument, CommandParser
from linktools.ssh import SSHClient


class Command(BaseCommand):

    @subcommand("reverse", pass_args=True)
    @subcommand_argument("--local-port")
    @subcommand_argument("--remote-port")
    def on_reverse(self, args: Namespace, local_port: int = 8000, remote_port: int = None):
        ssh_client: SSHClient = args.ssh_client
        reverse = ssh_client.reverse(remote_port=remote_port, forward_host="127.0.0.1", forward_port=local_port)

        self.logger.info(f"remote port: {reverse.remote_port}")
        time.sleep(20)
        self.logger.info("stop reverse")
        reverse.stop()
        time.sleep(1000)

    @subcommand("forward", pass_args=True)
    @subcommand_argument("--local-port")
    @subcommand_argument("--remote-port")
    def on_forward(self, args: Namespace, local_port: int = None, remote_port: int = 22):
        ssh_client: SSHClient = args.ssh_client
        forward = ssh_client.forward(forward_host="127.0.0.1", forward_port=remote_port, local_port=local_port)

        self.logger.info(f"local port: {forward.local_port}")
        time.sleep(20)
        self.logger.info("stop forward")
        forward.stop()
        time.sleep(1000)

    def init_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--host", type=str, required=True)
        parser.add_argument("--port", type=int, default=22)
        parser.add_argument("--username", type=str, default="root")
        parser.add_argument("--password", type=str)
        self.add_subcommands(parser)

    def run(self, args: Namespace) -> Optional[int]:
        ssh_client = SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect_with_pwd(
            hostname=args.host,
            port=args.port,
            username=args.username,
            password=args.password
        )
        setattr(args, "ssh_client", ssh_client)
        return self.run_subcommand(args)


command = Command()
if __name__ == '__main__':
    command.main()
