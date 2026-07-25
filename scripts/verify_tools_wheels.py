#!/usr/bin/env python3
"""Verify capability resource ownership and wheel install/uninstall isolation."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path


WHEEL_RESOURCES = {
    "linktools-": "linktools/assets/tools/linktools.json",
    "linktools_common-": "linktools/assets/tools/linktools-common.json",
    "linktools_mobile-": "linktools/assets/tools/linktools-mobile.json",
}


def wheel_files(directories):
    result = []
    for directory in directories:
        result.extend(sorted(Path(directory).glob("*.whl")))
    if len(result) != 3:
        raise SystemExit("expected exactly three capability wheels, got %d" % len(result))
    return result


def resource_for(wheel):
    for prefix, resource in WHEEL_RESOURCES.items():
        if wheel.name.startswith(prefix):
            return resource
    raise SystemExit("unknown wheel name: %s" % wheel.name)


def inspect_wheels(wheels):
    resources = {}
    tool_names = {}
    for wheel in wheels:
        resource = resource_for(wheel)
        with zipfile.ZipFile(str(wheel)) as archive:
            names = set(archive.namelist())
            assert resource in names, (wheel, resource)
            payload = json.loads(archive.read(resource).decode("utf-8"))
        resources[wheel] = resource
        tool_names[wheel] = set(payload["tools"])
    assert len(set(resources.values())) == 3, resources
    return tool_names


def python_path(environment):
    return Path(environment) / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed_tools(environment):
    code = "from linktools.core import environ; print('\\n'.join(sorted(environ.tools.keys())))"
    result = subprocess.run([str(python_path(environment)), "-c", code], check=True,
                            capture_output=True, text=True,
                            env=dict(os.environ, DEBUG="false", HOME=str(environment)))
    return set(result.stdout.splitlines())


def install_order(environment, wheels):
    try:
        venv.create(str(environment), with_pip=True, system_site_packages=True)
    except subprocess.CalledProcessError:
        if not shutil.which("uv"):
            raise
        shutil.rmtree(str(environment), ignore_errors=True)
        subprocess.run(["uv", "venv", str(environment)], check=True, stdout=subprocess.DEVNULL)
    if shutil.which("uv") and not python_path(environment).with_name("pip").exists():
        command = ["uv", "pip", "install", "--python", str(python_path(environment)), "--no-deps"]
    else:
        command = [str(python_path(environment)), "-m", "pip", "install", "--no-deps"]
    subprocess.run(command + [*[str(wheel) for wheel in wheels]],
                   check=True, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", nargs=3, help="three directories containing one wheel each")
    args = parser.parse_args()
    wheels = wheel_files(args.dist)
    tool_names = inspect_wheels(wheels)
    core_wheel = next(wheel for wheel in wheels if wheel.name.startswith("linktools-"))
    common_wheel = next(wheel for wheel in wheels if wheel.name.startswith("linktools_common-"))
    mobile_wheel = next(wheel for wheel in wheels if wheel.name.startswith("linktools_mobile-"))

    with tempfile.TemporaryDirectory(prefix="linktools-wheel-check-") as temporary:
        root = Path(temporary)
        environments = []
        for label, order in (("core-first", [core_wheel, common_wheel, mobile_wheel]),
                             ("mobile-first", [mobile_wheel, common_wheel, core_wheel])):
            environment = root / label
            install_order(environment, order)
            names = installed_tools(environment)
            environments.append((environment, names))
        assert environments[0][1] == environments[1][1], environments

        environment, before = environments[0]
        if shutil.which("uv") and not python_path(environment).with_name("pip").exists():
            uninstall = ["uv", "pip", "uninstall", "--python", str(python_path(environment))]
        else:
            uninstall = [str(python_path(environment)), "-m", "pip", "uninstall"]
        subprocess.run(uninstall + ["-y", "linktools-mobile"], check=True, stdout=subprocess.DEVNULL)
        after = installed_tools(environment)
        assert not (after & tool_names[mobile_wheel]), (after, tool_names[mobile_wheel])
        assert tool_names[core_wheel] | tool_names[common_wheel] <= after

        site_packages = next(environment.glob("lib/python*/site-packages"))
        assert (site_packages / "linktools/assets/tools/linktools.json").is_file()
        assert (site_packages / "linktools/assets/tools/linktools-common.json").is_file()
        print("wheel resource and install/uninstall checks passed")


if __name__ == "__main__":
    main()
