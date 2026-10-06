#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand package-owned checks into a CI execution plan."""

import json
import sys
import typing

from manage import get_modules, load_project_checks

PYTHON_VERSIONS = ("3.10", "3.x")


def package_checks(packages: "typing.Iterable[str]") -> "typing.List[typing.Dict[str, str]]":
    checks = []
    modules = get_modules()
    configs = {package: load_project_checks(package, modules[package]["path"]) for package in packages}
    bundles = {}
    for package, config in configs.items():
        pool = config.get("ci-pool")
        if pool is not None:
            if pool in configs and "ci-pool" not in configs[pool]:
                print("[-] CI pool %s conflicts with an independent package" % pool, file=sys.stderr)
                raise SystemExit(1)
        bundles.setdefault(pool or package, []).append(package)

    for package, bundle in bundles.items():
        config = configs[bundle[0]]
        groups = config.get("pytest", {}).get("groups", {"all": ()})
        for group in groups:
            checks.append({
                "package": package,
                "packages": " ".join(bundle),
                "group": group,
                # Core checks exercise every installed package's command entry points.
                "install": "" if "linktools" in bundle else " ".join(bundle),
                "name": "%s checks%s" % (
                    package, " (%s)" % group if group != "all" else "",
                ),
                "pytest-args": "-n 4 --dist=loadfile --capture=fd -rs --test-group=%s%s" % (
                    group, " --durations=50" if group != "all" else "",
                ),
            })
    return checks


if __name__ == "__main__":
    print(json.dumps({
        "python-versions": PYTHON_VERSIONS,
        "checks": package_checks(json.loads(sys.argv[1])),
    }))
