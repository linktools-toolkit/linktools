#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : TTools.py
@time    : 2018/12/11
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
import json
import subprocess
from argparse import ArgumentParser, Namespace
from typing import Optional, Type, List

from linktools import ToolError, DownloadError
from linktools.cli import BaseCommand, CommandMain
from linktools.cli.argparse import KeyValueAction


class Command(BaseCommand):
    """
    Execute tools directly from remote URLs
    """

    @property
    def main(self) -> CommandMain:
        return CommandMain(self, show_log_level=False, show_log_time=False)

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        return super().known_errors + [ToolError, DownloadError]

    def init_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("--set", action=KeyValueAction, nargs=1, dest="configs",
                            help="set the config of tool")

        group = parser.add_mutually_exclusive_group()
        group.add_argument("-c", "--config", action="store_true", default=False,
                           help="show the config of tool")
        group.add_argument("--download", action="store_true", default=False,
                           help="download tool files")
        group.add_argument("--clear", action="store_true", default=False,
                           help="clear tool files")
        group.add_argument("-d", "--daemon", action="store_true", default=False,
                           help="execute tools as a daemon")

        tool_names = sorted([tool.name for tool in self.environ.tools.values()])
        subparsers = parser.add_subparsers(metavar="TOOL", help=f"{{{','.join(tool_names)}}}")
        subparsers.required = True
        for tool_name in tool_names:
            tool_parser = subparsers.add_parser(tool_name, prefix_chars=chr(0), add_help=False)
            tool_parser.add_argument("tool_args", metavar="args", nargs="...")
            tool_parser.set_defaults(tool_name=tool_name)

    def run(self, args: Namespace) -> Optional[int]:

        tool_name, tool_args = args.tool_name, args.tool_args
        tool = self.environ.get_tool(tool_name, **(args.configs or {}))

        if args.config:
            self.logger.info(json.dumps(tool.config, indent=2, ensure_ascii=False))
            return 0

        elif args.download:
            tool.prepare()
            self.logger.info(f"Download tool files success: {tool.absolute_path}")
            return 0

        elif args.clear:
            tool.clear()
            self.logger.info(f"Clear tool files success")
            return 0

        elif args.daemon:
            process = tool.popen(
                *tool_args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=None,
            )
            try:
                return process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                return None

        else:
            process = tool.popen(*tool_args)
            return process.call()


command = Command()
if __name__ == "__main__":
    command.main()
