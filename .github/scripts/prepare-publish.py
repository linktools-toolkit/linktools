#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : perpare.py 
@time    : 2023/04/16
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
import json
import os
import re

import yaml

if __name__ == '__main__':
    root_path = os.path.abspath(os.path.join(__file__, "..", "..", "..", "src", "linktools"))

    version = os.environ["VERSION"]
    if version.startswith("v"):
        version = version[len("v"):]

    patten = re.compile(r"^__version__\s+=\s*\"\S*\"$")
    with open(os.path.join(root_path, "version.py"), "rt") as fd:
        file_data = fd.read()
    with open(os.path.join(root_path, "version.py"), "wt") as fd:
        for line in file_data.splitlines(keepends=True):
            fd.write(patten.sub(f"__version__ = \"{version}\"", line))

    with open(os.path.join(root_path, "assets", "tools.yml"), "rb") as fd:
        file_data = yaml.safe_load(fd)
    with open(os.path.join(root_path, "assets", "tools.json"), "wt") as fd:
        json.dump(file_data, fd, indent=2, ensure_ascii=True)
    os.remove(os.path.join(root_path, "assets", "tools.yml"))