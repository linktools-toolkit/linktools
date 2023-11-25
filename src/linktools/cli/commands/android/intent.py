#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : at_intent.py
@time    : 2018/12/04
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
import os
from argparse import ArgumentParser, Namespace
from typing import Optional

from linktools import utils
from linktools.cli import subcommand, subcommand_argument, AndroidCommand


class Command(AndroidCommand):
    """
    Common intent actions
    """

    def init_arguments(self, parser: ArgumentParser) -> None:
        subparser = self.add_subcommands(parser)
        subparser.required = True

    def run(self, args: Namespace) -> Optional[int]:
        return self.run_subcommand(args)

    @subcommand("setting", help="start setting activity", pass_args=True)
    def on_setting(self, args: Namespace):
        device = args.device_picker.pick()
        device.shell("am", "start", "--user", "0",
                     "-a", "android.settings.SETTINGS",
                     log_output=True)

    @subcommand("setting-dev", help="start development setting activity", pass_args=True)
    def on_setting_dev(self, args: Namespace):
        device = args.device_picker.pick()
        device.shell("am", "start", "--user", "0",
                     "-a", "android.settings.APPLICATION_DEVELOPMENT_SETTINGS",
                     log_output=True)

    @subcommand("setting-dev2", help="start development setting activity", pass_args=True)
    def on_setting_dev2(self, args: Namespace):
        device = args.device_picker.pick()
        device.shell("am", "start", "--user", "0",
                     "-a", "android.intent.action.View",
                     "com.android.settings/com.android.settings.DevelopmentSettings",
                     log_output=True)

    @subcommand("setting-app", help="start application setting activity (default: current running package)",
                pass_args=True)
    @subcommand_argument("package")
    def on_setting_app(self, args: Namespace, package: str = None):
        device = args.device_picker.pick()
        package = package if not utils.is_empty(package) else device.get_current_package()
        device.shell("am", "start", "--user", "0",
                     "-a", "android.settings.APPLICATION_DETAILS_SETTINGS",
                     "-d", "package:%s" % package,
                     log_output=True)

    @subcommand("setting-cert", help="install cert (need \'/data/local/tmp\' write permission)", pass_args=True)
    @subcommand_argument("path")
    def on_setting_cert(self, args: Namespace, path: str):
        device = args.device_picker.pick()
        remote_path = device.get_data_path("cert", os.path.basename(path))
        device.push(path, remote_path, log_output=True)
        device.shell("am", "start", "--user", "0",
                     "-n", "com.android.certinstaller/.CertInstallerMain",
                     "-a", "android.intent.action.VIEW",
                     "-t", "application/x-x509-ca-cert",
                     "-d", "file://%s" % remote_path,
                     log_output=True)

    @subcommand("install", help="install apk file (need \'/data/local/tmp\' write permission)", pass_args=True)
    @subcommand_argument("path")
    def on_install(self, args: Namespace, path: str):
        device = args.device_picker.pick()
        device.install(path,
                       opts=["-r", "-t", "-d", "-f"],
                       log_output=True)

    @subcommand("browser", help="start browser activity and jump to url", pass_args=True)
    @subcommand_argument("url", help="e.g. https://antiy.cn")
    def on_browser(self, args: Namespace, url: str):
        device = args.device_picker.pick()
        device.shell("am", "start", "--user", "0",
                     "-a", "android.intent.action.VIEW",
                     "-d", url,
                     log_output=True)


command = Command()
if __name__ == "__main__":
    command.main()
