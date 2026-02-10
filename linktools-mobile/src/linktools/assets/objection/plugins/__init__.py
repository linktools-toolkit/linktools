#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : __init__.py.py 
@time    : 2022/11/20
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
import functools
from typing import Any

import click
from objection.state.app import app_state
from objection.utils.plugin import Plugin

from linktools.core import environ
from linktools.capabilities.mobile import __capability__
from linktools.mobile.frida import FridaEvalCode, FridaScript, FridaScriptHandler, FridaSession

__description__ = f"{environ.name} plugin"


class LinktoolsPlugin(Plugin, FridaScriptHandler):

    def __init__(self, ns):
        self.script_path = __capability__.get_asset_path("frida.js")

        super().__init__(__file__, ns, {
            'meta': f'{environ.name}',
            'commands': {
                'eval': {
                    'meta': 'eval code',
                    'exec': self.eval_code
                }
            }
        })

        self.inject()

        if app_state.should_debug():
            self.eval_code([
                "Log.setLevel(Log.DEBUG);"
            ])

    def eval_code(self, args: list):
        self.api.load_scripts([
            FridaEvalCode(arg).as_dict() for arg in args
        ])

    @functools.cached_property
    def _frida_script(self) -> FridaScript:
        return FridaScript(
            FridaSession(self.session),
            self.script
        )

    def on_message_handler(self, message, data):
        return self.on_script_message(
            self._frida_script, message, data
        )

    def on_script_log(self, script: FridaScript, level: str, message: Any, data: Any):
        if level == self.LogLevel.INFO:
            click.secho(f"[{level}] ({namespace}) {message}")
        elif level == self.LogLevel.WARNING:
            click.secho(f"[{level}] ({namespace}) {message}", fg="yellow")
        elif level == self.LogLevel.ERROR:
            click.secho(f"[{level}] ({namespace}) {message}", fg="red")
        elif app_state.should_debug():
            click.secho(f"[{level}] ({namespace}) {message}", dim=True)


namespace = 'lt'
plugin = LinktoolsPlugin
