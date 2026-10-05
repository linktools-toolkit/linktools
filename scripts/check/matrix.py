#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand discovered packages into independently executable CI checks."""

import json
import sys
import typing


def package_checks(packages: "typing.Iterable[str]") -> "typing.List[typing.Dict[str, str]]":
    checks = []
    for package in packages:
        groups = ("evaluation", "runtime") if package == "linktools-ai" else ("all",)
        for group in groups:
            checks.append({
                "package": package,
                "group": group,
                "name": "%s checks%s" % (
                    package, " (%s)" % group if group != "all" else "",
                ),
            })
    return checks


if __name__ == "__main__":
    print(json.dumps(package_checks(json.loads(sys.argv[1]))))
