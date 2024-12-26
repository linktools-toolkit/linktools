#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : struct.py
@time    : 2019/01/11
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

from .. import utils


class App:

    def __init__(self, obj: dict):
        self.bundle_id = utils.get_item(obj, "CFBundleIdentifier", type=str, default="")
        self.name = utils.get_item(obj, "CFBundleName", type=str, default="")
        self.short_version = utils.get_item(obj, "CFBundleVersion", type=str, default="")
        self.version = utils.get_item(obj, "CFBundleShortVersionString", type=str, default="")
        self.container = utils.get_item(obj, "Container", type=str, default="")
        self.entitlements = utils.get_item(obj, "Entitlements", type=dict, default="")

    def __repr__(self):
        return f"App<{self.bundle_id}>"


class Process:

    def __init__(self, obj: dict):
        self.pid = utils.get_item(obj, "Pid", type=int, default=0)
        self.name = utils.get_item(obj, "Name", type=str, default=0)
        self.real_app_name = utils.get_item(obj, "RealAppName", type=str, default="")
        self.is_application = utils.get_item(obj, "IsApplication", type=bool, default=False)
        self.start_date = utils.get_item(obj, "StartDate", type=str, default="")

    def __repr__(self):
        return f"Process<{self.name}>"
