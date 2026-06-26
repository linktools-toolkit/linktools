#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

from ._utils import get_environ, get_logger

_user_agent = None


def user_agent(style=None) -> str:
    global _user_agent

    if _user_agent is None:
        from linktools.references.fake_useragent import UserAgent

        class _UserAgent(UserAgent):

            def __init__(self):
                environ = get_environ()
                super().__init__(
                    path=environ.get_path("assets", "browsers.json"),
                    fallback=environ.get_config("DEFAULT_USER_AGENT", type=str),
                )

        _user_agent = _UserAgent()

    try:
        if style:
            return _user_agent[style]
        return _user_agent.random
    except Exception as e:
        get_logger().debug(f"fetch user agent error: {e}")

    return _user_agent.fallback


def make_url(scheme: str, host: str, port: int, *paths: str, queries=None) -> str:
    url = f"{scheme}://{host}"
    if port is not None:
        if (scheme == "http" and port != 80) or (scheme == "https" and port != 443):
            url += f":{port}"
    return join_url(url, *paths, queries=queries)


def join_url(url: str, *paths: str, queries=None) -> str:
    from urllib import parse

    result = url
    for path in paths:
        if path:
            result = result.rstrip("/") + "/" + path.lstrip("/")

    if queries:
        query_list = []
        for key, value in queries.items():
            if isinstance(value, (list, tuple)):
                query_list.extend((key, v) for v in value)
            else:
                query_list.append((key, value))
        result = result + "?" + parse.urlencode(query_list)

    return result


def guess_file_name(url: str) -> str:
    from urllib import parse
    if not url:
        return ""
    try:
        return os.path.split(parse.urlparse(url).path)[1]
    except:
        return ""


def _parseparam(s):
    while s[:1] == ";":
        s = s[1:]
        end = s.find(";")
        while end > 0 and (s.count('"', 0, end) - s.count('\\"', 0, end)) % 2:
            end = s.find(";", end + 1)
        if end < 0:
            end = len(s)
        f = s[:end]
        yield f.strip()
        s = s[end:]


def parse_header(line):
    parts = _parseparam(";" + line)
    key = parts.__next__()
    pdict = {}
    for p in parts:
        i = p.find("=")
        if i >= 0:
            name = p[:i].strip().lower()
            value = p[i + 1:].strip()
            if len(value) >= 2 and value[0] == value[-1] == '"':
                value = value[1:-1]
                value = value.replace("\\\\", "\\").replace('\\"', '"')
            pdict[name] = value
    return key, pdict


def parse_cookie(cookie: str) -> "dict[str, str]":
    cookies = {}
    for item in cookie.split(";"):
        key_value = item.split("=", 1)
        cookies[key_value[0].strip()] = key_value[1].strip() if len(key_value) > 1 else ""
    return cookies
