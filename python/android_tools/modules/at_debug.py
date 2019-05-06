#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : at_debug.py 
@time    : 2019/04/22
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
from android_tools import utils
from android_tools.adb import Device, AdbError
from android_tools.argparser import AdbArgumentParser


def main():
    parser = AdbArgumentParser(description='debugger')

    group = parser.add_argument_group(title="common arguments")
    group.add_argument('package', action='store', default=None,
                       help='regular expression')
    group.add_argument('activity', action='store', default=None,
                       help='regular expression')
    group.add_argument('-p', '--port', action='store', type=int, default=8700,
                       help='fetch all apps')

    args = parser.parse_args()
    device = Device(args.parse_adb_serial())

    device.shell("am", "force-stop", args.package, capture_output=False)
    device.shell("am", "start", "-D", "-n", "{}/{}".format(args.package, args.activity), capture_output=False)

    pid = utils.int(device.shell("top", "-n", "1", "|", "grep", args.package).split()[0])
    device.exec("forward", "tcp:{}".format(args.port), "jdwp:{}".format(pid), capture_output=False)

    data = input("jdb connect? [Y/n]: ").strip()
    if data in ["", "Y", "y"]:
        utils.exec("jdb", "-connect", "com.sun.jdi.SocketAttach:hostname=127.0.0.1,port={}".format(args.port),
                   capture_output=False)


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, EOFError, AdbError) as e:
        print(e)
