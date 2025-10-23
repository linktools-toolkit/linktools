#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : ssh.py 
@time    : 2022/11/27
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
import shutil
from typing import Optional, Type, List

import paramiko
from paramiko.ssh_exception import SSHException

from linktools import utils
from linktools.cli import CommandParser, CommandMain, IOSCommand, IOSNamespace
from linktools.ssh import SSHClient


class Command(IOSCommand):
    """
    Remotely login to jailbroken iOS devices using the OpenSSH client
    """

    @property
    def main(self) -> CommandMain:
        return CommandMain(self, show_log_level=False, show_log_time=False)

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        return super().known_errors + [SSHException]

    def init_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("-u", "--username", action="store", default="root",
                            help="iOS ssh username (default: root)")
        parser.add_argument("-p", "--port", action="store", type=int, default=22,
                            help="iOS ssh port (default: 22)")
        parser.add_argument("ssh_args", nargs="...", help="ssh args")

    def run(self, args: IOSNamespace) -> Optional[int]:
        device = args.device_selector.select()

        local_port = utils.get_free_port()
        with device.forward(local_port, args.port):
            ssh = shutil.which("ssh")
            if ssh:
                option_args = [
                    "-o", "StrictHostKeyChecking=no",
                    "-o", f"UserKnownHostsFile={os.devnull}",
                ]
                process = utils.popen(
                    ssh,
                    *option_args,
                    f"{args.username}@127.0.0.1", "-p", local_port,
                    *args.ssh_args,
                    capture_output=False,
                )
                return process.call()

            with SSHClient() as client:
                client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect_with_pwd("localhost", port=local_port, username=args.username)
                client.open_shell(*args.ssh_args)
                return 0


command = Command()
if __name__ == "__main__":
    command.main()
