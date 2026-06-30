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

from ._common import (
    ignore_errors,
    cast, cast_int as int, cast_bool as bool,
    coalesce, is_contain, is_empty,
    get_item, pop_item, get_list_item,
    make_uuid, random_string,
    gzip_compress,
    parse_version, get_char_width,
    let, also,
    list2cmdline, cmdline2list,
)

from ._hash import (
    get_hash, get_hash_ident, get_file_hash, get_md5, get_file_md5,
)

from ._files import (
    is_sub_path, join_path, read_file, write_file, remove_file, clear_directory,
)

from ._urls import (
    make_url, join_url, parse_header, parse_cookie, guess_file_name, user_agent,
)
