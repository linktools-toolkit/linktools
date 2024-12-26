#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author  : Hu Ji
@file    : __init__.py.py 
@time    : 2022/11/19
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

from ._utils import (
    ignore_error,
    cast, cast_int as int, cast_bool as bool,
    coalesce, is_contain, is_empty,
    get_item, pop_item, get_list_item,
    get_hash, get_file_hash, get_md5, get_file_md5,
    make_uuid, gzip_compress,
    is_sub_path, join_path, read_file, write_file, remove_file, clear_directory,
    get_lan_ip, get_wan_ip,
    parse_version, get_char_width,
    make_url, parse_header, parser_cookie, guess_file_name, user_agent,
    get_interpreter, get_system, get_machine, get_user, get_uid, get_gid, get_shell_path,
    is_unix_like, is_windows,
    import_module, import_module_file,
    get_derived_type, lazy_load, lazy_iter, lazy_raise,
    wait_event, wait_thread, wait_process,
)

from ._subprocess import (
    Process, popen, popen as create_process,
    list2cmdline, cmdline2list,
)

from ._port import (
    is_port_free, get_free_port,
    NoFreePortFoundError,
)
