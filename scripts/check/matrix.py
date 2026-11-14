#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand package-owned checks into a CI execution plan."""

import json
import sys
import typing

from manage import get_modules, load_project_checks

PYTHON_VERSIONS = ("3.11", "3.x")


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
        pytest = config.get("pytest", {})
        groups = pytest.get("groups", {"all": ()})
        for group in groups:
            count = pytest.get("shards", {}).get(group, 1)
            for index in range(1, count + 1):
                shard = "%d/%d" % (index, count) if count > 1 else ""
                label = "%s shard %s" % (group, shard) if shard else group
                checks.append({
                    "package": package,
                    "packages": " ".join(bundle),
                    "group": group,
                    "shard": shard,
                    # Core checks exercise every installed package's command entry points.
                    "install": "" if "linktools" in bundle else " ".join(bundle),
                    "name": "%s checks%s" % (
                        " + ".join(bundle), " (%s)" % label if label != "all" else "",
                    ),
                    "pytest-args": "-n 4 --dist=loadfile --capture=fd -r%s --test-group=%s%s%s%s" % (
                        "a" if group != "all" else "s", group,
                        " --test-shard=" + shard if shard else "",
                        " --durations=50",
                        " --tb=long -o verbosity_assertions=2 -o faulthandler_timeout=90" if group != "all" else "",
                    ),
                })
    return checks


if __name__ == "__main__":
    print(json.dumps({
        "python-versions": PYTHON_VERSIONS,
        "checks": package_checks(json.loads(sys.argv[1])),
    }))
