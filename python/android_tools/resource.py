#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : resource.py 
@time    : 2018/12/01
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

from .decorator import singleton
from .utils import utils


@singleton
class Resource(object):
    _res_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resource")

    def __init__(self):
        self.config = None

    def _get_path(self, *paths: [str]):
        return os.path.join(self._res_path, *paths)

    def get_config(self, path: str, *key: [str]):
        with open(self._get_path("config", path), "rt") as fd:
            config = json.load(fd)
        return utils.get_item(config, *key)

    def get_persist_path(self, *paths: [str]):
        return self._get_path("persist", *paths)

    def get_storage_path(self, *paths: [str], create_dir: bool = False, create_file: bool = False):
        path = os.path.join(self._res_path, "storage", *paths)
        if create_dir or create_file:
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        if create_file:
            if not os.path.exists(path):
                open(path, 'a').close()
        return path

    def config_getter(self, path: str, *key: [str]):
        configs = {}

        # noinspection PyUnusedLocal
        def decorator(fn):
            # noinspection PyUnusedLocal
            def wrapper(*args, **kwargs):
                if path not in configs:
                    with open(self._get_path("config", path), "rt") as fd:
                        configs[path] = json.load(fd)
                return utils.get_item(configs[path], *key)
            return wrapper
        return decorator


resource = Resource()
