#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : __main__.py 
@time    : 2022/12/13
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

from .cli import commands, BaseCommandGroup, CommandParser
from .cli import iter_module_commands, iter_entry_point_commands
from .metadata import __ep_scripts__


class Command(BaseCommandGroup):

    def init_arguments(self, parser: CommandParser) -> None:
        self.add_subcommands(
            parser=parser,
            target=(
                iter_module_commands(commands, onerror="warn"),
                iter_entry_point_commands(__ep_scripts__, onerror="warn")
            ),
            sort=True
        )


command = Command()
if __name__ == '__main__':
    command.main()
