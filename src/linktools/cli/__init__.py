#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : __init__.py.py 
@time    : 2023/01/14
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

from . import argparse

from .command import \
    BaseCommand, BaseCommandGroup, CommandError, \
    SubCommand, SubCommandGroup, SubCommandWrapper, \
    subcommand, subcommand_argument, SubCommandError, NotFoundSubCommand, \
    iter_module_commands, iter_entry_point_commands, \
    CommandMain, CommandParser

from .device import \
    DeviceCommandMixin, AndroidCommandMixin, IOSCommandMixin, \
    AndroidCommand, IOSCommand
