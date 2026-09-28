#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    get_environ, get_logger,
)

from ._hash import (
    get_hash, get_hash_ident, get_file_hash, get_md5, get_file_md5, verify_file,
)

from ._files import (
    is_sub_path, join_path, safe_join, ensure_within, read_file, write_file,
    atomic_write, atomic_replace, remove_file, clear_directory, safe_remove,
    safe_rmtree,
)

from ._urls import (
    make_url, join_url, parse_header, parse_cookie, guess_file_name, user_agent,
)

from ._archive import safe_extract
