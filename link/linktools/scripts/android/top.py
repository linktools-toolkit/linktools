#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : at_top_app.py
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
   /  oooooooooooooooo  .o.  oooo /,   \,"-----------
  / ==ooooooooooooooo==.o.  ooo= //   ,`\--{)B     ,"
 /_==__==========__==_ooo__ooo=_/'   /___________,"
"""

import datetime
import sys

from linktools import utils, logger
from linktools.android import AdbError, AndroidArgumentParser
from linktools.decorator import entry_point


@entry_point(known_errors=[AdbError])
def main():
    parser = AndroidArgumentParser(description='show current running app\'s basic information')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-p', '--package', action='store_const', const=True, default=False,
                       help='show current running package name')
    group.add_argument('-a', '--activity', action='store_const', const=True, default=False,
                       help='show current running activity name')
    group.add_argument('--path', action='store_const', const=True, default=False,
                       help='show current running package path')
    group.add_argument('--kill', action='store_const', const=True, default=False,
                       help='kill current running package')
    group.add_argument('--apk', metavar='DEST', action='store', type=str, nargs='?', default=".",
                       help='pull current running apk file')
    group.add_argument('--screen', metavar='DEST', action='store', type=str, nargs='?', default=".",
                       help='capture screen and pull file')

    args = parser.parse_args()
    device = args.parse_device()

    if args.package:
        logger.message(device.get_current_package())
    elif args.activity:
        logger.message(device.get_current_activity())
    elif args.path:
        logger.message(device.get_apk_path(device.get_current_package()))
    elif args.kill:
        device.shell("am", "force-stop", device.get_current_package(), capture_output=False)
    elif "--apk" in sys.argv:
        package_name = device.get_current_package()
        logger.message("get current running package: {}".format(package_name))
        package = utils.get_item(device.get_packages(package_name, basic_info=True), 0)
        if package is not None:
            logger.message("get current running package path: {}".format(package.source_dir))
            path = device.get_storage_path("{}_{}.apk".format(package.name, package.version_name))
            dest = args.apk if not utils.is_empty(args.apk) else "."
            device.shell("mkdir", "-p", device.get_storage_path(), capture_output=False)
            device.shell("cp", package.source_dir, path, capture_output=False)
            device.pull(path, dest, capture_output=False)
            device.shell("rm", path)
    elif "--screen" in sys.argv:
        now = datetime.datetime.now()
        path = device.get_storage_path("screenshot-" + now.strftime("%Y-%m-%d-%H-%M-%S") + ".png")
        dest = args.screen if not utils.is_empty(args.screen) else "."
        device.shell("mkdir", "-p", device.get_storage_path(), capture_output=False)
        device.shell("screencap", "-p", path, capture_output=False)
        device.pull(path, dest, capture_output=False)
        device.shell("rm", path)
    else:
        package = device.get_current_package()
        logger.message("package:  ", package)
        logger.message("activity: ", device.get_current_activity())
        logger.message("path:     ", device.get_apk_path(package))


if __name__ == '__main__':
    main()
