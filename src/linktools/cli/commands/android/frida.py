#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : at_frida.py
@time    : 2018/11/25
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
import re
from argparse import ArgumentParser, Namespace
from typing import Optional, List, Type

from linktools import utils, DownloadError
from linktools.cli import CommandError, AndroidCommand, CommandMain
from linktools.cli.argparse import range_type, KeyValueAction, BooleanOptionalAction
from linktools.frida import FridaApplication, FridaShareScript, FridaScriptFile, FridaEvalCode, FridaAndroidServer


class Command(AndroidCommand):
    """
    Use Frida for dynamic analysis on rooted Android devices
    """

    @property
    def main(self) -> CommandMain:
        return CommandMain(self, show_log_level=True, show_log_time=True)

    @property
    def known_errors(self) -> List[Type[BaseException]]:
        return super().known_errors + [DownloadError]

    def init_arguments(self, parser: ArgumentParser) -> None:
        parser.add_argument("-p", "--package", action="store", default=None,
                            help="target package (default: frontmost application)")
        parser.add_argument("--spawn", action="store_true", default=False,
                            help="inject after spawn (default: false)")

        parser.add_argument("-P", "--parameters",
                            action=KeyValueAction, nargs="+", dest="user_parameters",
                            help="user script parameters")

        parser.add_argument("-l", "--load", metavar="SCRIPT",
                            action="append", dest="user_scripts", default=[],
                            type=lambda o: FridaScriptFile(o),
                            help="load user script")
        parser.add_argument("-e", "--eval", metavar="CODE", action="append", dest="user_scripts",
                            type=lambda o: FridaEvalCode(o),
                            help="evaluate code")
        parser.add_argument("-c", "--codeshare", metavar="URL", action="append", dest="user_scripts",
                            type=lambda o: FridaShareScript(o, cached=False),
                            help="load share script url")

        parser.add_argument("-a", "--auto-start", action="store_true", default=False,
                            help="automatically start when all processes exits")

        parser.add_argument("--serve", action=BooleanOptionalAction, default=True,
                            help="serve frida server (default: true)")
        parser.add_argument("--local-port", metavar="PORT", action="store", dest="local_port",
                            type=range_type(1, 65536), default=None,
                            help="local frida port (default: unused port)")
        parser.add_argument("--remote-port", metavar="PORT", action="store", dest="remote_port",
                            type=range_type(1, 65536), default=27042,
                            help="remote frida port (default: 27042)")

        parser.add_argument("--spawn-gating", action=BooleanOptionalAction, default=True,
                            help="enable spawn gating (default: true)")
        parser.add_argument("--child-gating", action=BooleanOptionalAction, default=False,
                            help="enable child gating (default: false)")

        parser.add_argument("--redirect-address", metavar="ADDRESS", action="store", dest="redirect_address",
                            type=str,
                            help="redirect traffic to target address (default: localhost)")
        parser.add_argument("--redirect-port", metavar="PORT", action="store", dest="redirect_port",
                            type=range_type(1, 65536),
                            help="redirect traffic to target port (default: 8080)")

    def run(self, args: Namespace) -> Optional[int]:

        logger = self.logger
        user_parameters = args.user_parameters
        user_scripts = args.user_scripts

        device = args.device_picker.pick()
        package = args.package

        class Application(FridaApplication):

            def on_session_detached(self, session, reason, crash) -> None:
                logger.info(f"{session} detached, reason={reason}")
                if reason in ("connection-terminated", "device-lost"):
                    self.signal_stop()
                elif len(self.sessions) == 0:
                    if args.auto_start:
                        app.spawn(package)

        server = FridaAndroidServer(
            device=device,
            local_port=args.local_port or utils.get_free_port(),
            remote_port=args.remote_port,
            serve=args.serve
        )

        with server:
            # 如果没有填包名，则找到顶层应用
            if utils.is_empty(package):
                target_app = server.get_frontmost_application()
                if target_app is None:
                    raise CommandError("Unknown frontmost application")
                package = target_app.identifier

            logger.info(f"Frida inject target application: {package}")

            app = Application(
                server,
                target_identifiers=rf"^{re.escape(package)}($|:)",
                user_parameters=user_parameters,
                user_scripts=user_scripts,
                enable_spawn_gating=args.spawn_gating,
                enable_child_gating=args.child_gating
            )

            if args.spawn:
                # 打开进程后注入
                app.spawn(package)

            elif app.inject_all():
                # 注入所有进程进程
                pass

            elif args.auto_start:
                # 进程不存在，打开进程后注入
                app.spawn(package)

            if args.redirect_address or args.redirect_port:
                # 如果需要重定向到本地端口
                address = args.redirect_address
                port = args.redirect_port or 8080
                uid = device.get_uid(package)
                with device.redirect(address, port, uid):
                    return app.run()
            else:
                return app.run()


command = Command()
if __name__ == "__main__":
    command.main()
