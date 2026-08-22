#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import subprocess
import sys
import typing

import tomlkit
from tomlkit.items import Array

MODULE_NAME = "linktools"
TEMPLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))

_PROJECT_CHECKS = {
    "linktools": {
        "python": (3, 6),
        "tests": ("tests/core", "tests/cli"),
    },
    "linktools-common": {
        "python": (3, 6),
        "tests": ("tests/common",),
    },
    "linktools-mobile": {
        "python": (3, 6),
        "tests": (),
    },
    "linktools-cntr": {
        "python": (3, 6),
        "tests": ("tests/cntr",),
    },
    "linktools-ai": {
        "python": (3, 10),
        "tests": ("tests/ai",),
    },
}

__missing__ = object()


class LazyChoices(typing.Iterable):

    def __init__(self, func: typing.Callable[..., typing.Iterable], *args: typing.Any, **kwargs: typing.Any) -> None:
        self._data = __missing__
        self._fn = func
        self._args = args
        self._kwargs = kwargs

    def _load(self) -> typing.Iterable:
        result = self._data
        if result is __missing__:
            result = self._data = self._fn(*self._args, **self._kwargs)
        return typing.cast(typing.Iterable, result)

    def __iter__(self) -> typing.Iterator:
        return iter(self._load())

    def __contains__(self, item: typing.Any) -> bool:
        if item == argparse.SUPPRESS:
            return True
        if isinstance(item, (list, tuple)) and len(item) == 0:
            return True
        return item in self._load()


def get_modules() -> "typing.Dict[str, typing.Dict[str, str]]":
    modules = {}
    for name in sorted(os.listdir(PROJECT_PATH)):
        path = os.path.join(PROJECT_PATH, name)
        if not os.path.isdir(path):
            continue
        if name == MODULE_NAME:
            modules[MODULE_NAME] = {
                "name": "",
                "module": MODULE_NAME,
                "path": path,
            }
            continue
        if not name.startswith(MODULE_NAME):
            continue
        match = re.match(r"%s[-_](.+)" % re.escape(MODULE_NAME), name)
        if not match:
            continue
        simple_name = match.group(1)
        module_name = "%s-%s" % (MODULE_NAME, simple_name)
        modules[module_name] = {
            "name": simple_name,
            "module": module_name,
            "path": path,
        }
    return modules


def update_toml_recursive(source_data: typing.Any, target_data: typing.Any, **format_kwargs: typing.Any) -> None:
    if isinstance(source_data, dict):
        for key, value in source_data.items():
            if isinstance(value, (dict, list)):
                if key not in target_data:
                    target_data[key] = tomlkit.table() if isinstance(value, dict) else tomlkit.array()
                update_toml_recursive(value, target_data[key], **format_kwargs)
            elif isinstance(value, str):
                target_data[key] = value.format(**format_kwargs)
            else:
                target_data[key] = value
    elif isinstance(source_data, list):
        new_array = tomlkit.array()
        for item in source_data:
            if isinstance(item, str):
                new_array.append(item.format(**format_kwargs))
            elif isinstance(item, (dict, list)):
                temp_container = tomlkit.inline_table() if isinstance(item, dict) else tomlkit.array()
                update_toml_recursive(item, temp_container, **format_kwargs)
                new_array.append(temp_container)
            else:
                new_array.append(item)
        if isinstance(target_data, Array):
            target_data.clear()
            for value in new_array:
                target_data.append(value)


def sync_pyproject_toml(source_path: str, target_path: str, **format_vars: typing.Any) -> None:
    print("[+] Syncing pyproject.toml: %s -> %s" % (source_path, target_path))
    with open(source_path, "r", encoding="utf-8") as source_file:
        source_doc = tomlkit.parse(source_file.read())
    try:
        with open(target_path, "r", encoding="utf-8") as target_file:
            target_doc = tomlkit.parse(target_file.read())
    except FileNotFoundError:
        target_doc = tomlkit.document()
    update_toml_recursive(source_doc, target_doc, **format_vars)
    with open(target_path, "w", encoding="utf-8") as target_file:
        target_file.write(tomlkit.dumps(target_doc))


def sync_project_file(source_path: str, target_path: str, exist_ok: bool = True) -> None:
    if not os.path.exists(target_path) or exist_ok:
        print("[+] Syncing project file: %s -> %s" % (source_path, target_path))
        shutil.copy2(source_path, target_path)
    else:
        print("[-] Skipping existing file: %s" % target_path)


def _add_project_argument(parser: argparse.ArgumentParser, modules: "typing.Dict[str, typing.Dict[str, str]]") -> None:
    parser.add_argument(
        "module",
        choices=LazyChoices(sorted, modules.keys()),
        nargs="*",
        help="Project modules",
    )


def build_parser() -> argparse.ArgumentParser:
    modules = get_modules()
    parser = argparse.ArgumentParser(
        description="Project management tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.required = True

    init_parser = subparsers.add_parser("init", help="Initialize new project metadata")
    _add_project_argument(init_parser, modules)
    init_parser.set_defaults(func=handle_init)

    install_parser = subparsers.add_parser("install", help="Install project modules")
    _add_project_argument(install_parser, modules)
    install_parser.add_argument("-e", "--editable", action="store_true", help="Install in editable mode")
    install_parser.add_argument("--no-isolation", action="store_true", help="Disable build isolation")
    install_parser.set_defaults(func=handle_install)

    check_parser = subparsers.add_parser("check", help="Run source and test release gates")
    _add_project_argument(check_parser, modules)
    check_parser.set_defaults(func=handle_check)

    build_command = subparsers.add_parser("build", help="Build project modules")
    _add_project_argument(build_command, modules)
    build_command.set_defaults(func=handle_build)

    verify_parser = subparsers.add_parser("verify", help="Verify built release artifacts")
    _add_project_argument(verify_parser, modules)
    verify_parser.set_defaults(func=handle_verify)

    clean_parser = subparsers.add_parser("clean", help="Clean project build files")
    _add_project_argument(clean_parser, modules)
    clean_parser.set_defaults(func=handle_clean)
    return parser


def _selected_modules(args: argparse.Namespace) -> "typing.Tuple[str, ...]":
    if args.module:
        return tuple(dict.fromkeys(args.module))
    return tuple(_PROJECT_CHECKS.keys())


def _validate_project_registration() -> None:
    discovered = set(get_modules())
    registered = set(_PROJECT_CHECKS)
    missing = sorted(discovered - registered)
    stale = sorted(registered - discovered)
    if not missing and not stale:
        return
    if missing:
        print("[-] Unregistered projects: %s" % ", ".join(missing), file=sys.stderr)
    if stale:
        print("[-] Registered projects not found: %s" % ", ".join(stale), file=sys.stderr)
    raise SystemExit(1)


def _expected_requires_python(project: str) -> str:
    version = _PROJECT_CHECKS[project]["python"]
    return ">=%d.%d" % (version[0], version[1])


def _read_requires_python(project: str) -> str:
    path = os.path.join(PROJECT_PATH, project, "pyproject.toml")
    with open(path, "r", encoding="utf-8") as file:
        data = tomlkit.parse(file.read())
    value = data.get("project", {}).get("requires-python")
    return value if isinstance(value, str) else ""


def _validate_requires_python(projects: "typing.Tuple[str, ...]") -> None:
    for project in projects:
        actual = _read_requires_python(project)
        expected = _expected_requires_python(project)
        if actual != expected:
            print(
                "[-] %s requires-python is %r, expected %r" % (project, actual, expected),
                file=sys.stderr,
            )
            raise SystemExit(1)


def _validate_interpreter(projects: "typing.Tuple[str, ...]") -> None:
    required = max(_PROJECT_CHECKS[project]["python"] for project in projects)
    if sys.version_info[:2] >= required:
        return
    owners = [project for project in projects if _PROJECT_CHECKS[project]["python"] == required]
    print(
        "[-] Python %d.%d is required by %s; current interpreter is %d.%d"
        % (required[0], required[1], ", ".join(owners), sys.version_info[0], sys.version_info[1]),
        file=sys.stderr,
    )
    raise SystemExit(1)


def _check_environment() -> "typing.Dict[str, str]":
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _run_check(command: "typing.Sequence[str]", environment: "typing.Dict[str, str]") -> None:
    print("[+] Running: %s" % " ".join(command))
    subprocess.check_call(list(command), cwd=PROJECT_PATH, env=environment)


def _run_python36_gate(projects: "typing.Tuple[str, ...]", environment: "typing.Dict[str, str]") -> None:
    compatible = [project for project in projects if _PROJECT_CHECKS[project]["python"] == (3, 6)]
    if not compatible:
        return
    scanner = os.path.join(PROJECT_PATH, "scripts", "check", "python36.py")
    paths = [os.path.join(PROJECT_PATH, "manage.py"), scanner]
    paths.extend(os.path.join(PROJECT_PATH, project, "src") for project in compatible)
    _run_check([sys.executable, scanner] + paths, environment)


def _run_ai_ruff(environment: "typing.Dict[str, str]") -> None:
    ruff = shutil.which("ruff")
    if not ruff:
        print("[-] ruff is required; run `python manage.py install --editable` first", file=sys.stderr)
        raise SystemExit(1)
    _run_check(
        [
            ruff,
            "check",
            "--no-cache",
            "scripts/check/ai",
            "linktools-ai/src/linktools/ai",
            "linktools-ai/src/linktools/commands/ai",
            "tests/ai",
        ],
        environment,
    )


def _pytest_paths(args: argparse.Namespace, projects: "typing.Tuple[str, ...]") -> "typing.Tuple[str, ...]":
    if not args.module:
        return ("tests",)
    result = []
    seen = set()
    for project in projects:
        for path in _PROJECT_CHECKS[project]["tests"]:
            if path not in seen:
                seen.add(path)
                result.append(path)
    return tuple(result)


def handle_init(args: argparse.Namespace) -> None:
    for name, info in get_modules().items():
        if name == MODULE_NAME:
            continue
        if not args.module or name in args.module:
            print("[+] Initializing project: %s" % name)
            sync_pyproject_toml(
                os.path.join(TEMPLATE_PATH, "pyproject.template"),
                os.path.join(info["path"], "pyproject.toml"),
                **info
            )
            sync_project_file(
                os.path.join(TEMPLATE_PATH, "capability.jinja2"),
                os.path.join(info["path"], "capability.jinja2"),
                exist_ok=True,
            )
            sync_project_file(
                os.path.join(TEMPLATE_PATH, "MANIFEST.in"),
                os.path.join(info["path"], "MANIFEST.in"),
                exist_ok=True,
            )
            sync_project_file(
                os.path.join(TEMPLATE_PATH, "requirements.yml"),
                os.path.join(info["path"], "requirements.yml"),
                exist_ok=False,
            )


def handle_install(args: argparse.Namespace) -> None:
    paths = []
    for name, info in get_modules().items():
        if not args.module or name in args.module:
            print("[+] Adding module to install list: %s" % name)
            paths.append(info["path"])
    pip_args = [sys.executable, "-m", "pip", "install"]
    for path in paths:
        if args.editable:
            pip_args.append("-e")
        pip_args.append(path)
    if args.no_isolation:
        pip_args.append("--no-build-isolation")
    print("[+] Running pip install with arguments: %s" % " ".join(pip_args))
    subprocess.check_call(pip_args)


def handle_check(args: argparse.Namespace) -> None:
    _validate_project_registration()
    projects = _selected_modules(args)
    _validate_requires_python(projects)
    _validate_interpreter(projects)
    environment = _check_environment()
    _run_python36_gate(projects, environment)
    if "linktools-ai" in projects:
        _run_ai_ruff(environment)
    tests = _pytest_paths(args, projects)
    if tests:
        _run_check([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + list(tests), environment)
    else:
        for project in projects:
            if not _PROJECT_CHECKS[project]["tests"]:
                print("[+] %s: no dedicated test suite" % project)
    print("[+] Check passed: %s" % ", ".join(projects))


def handle_build(args: argparse.Namespace) -> None:
    version = os.environ.get("VERSION")
    for name, info in get_modules().items():
        if not args.module or name in args.module:
            print("[+] Building project: %s, path: %s" % (name, info["path"]))
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--sdist",
                    "--wheel",
                    "--outdir",
                    os.path.join(PROJECT_PATH, "dist"),
                    info["path"],
                ],
                cwd=PROJECT_PATH,
            )
            if version is not None:
                print("[+] Setting version for project: %s to %s" % (name, version))
                with open(os.path.join(info["path"], ".version"), "wt", encoding="utf-8") as file:
                    file.write(version)


def handle_verify(args: argparse.Namespace) -> None:
    _validate_project_registration()
    projects = _selected_modules(args)
    command = [sys.executable, os.path.join(PROJECT_PATH, "scripts", "verify.py")]
    command.extend(projects)
    subprocess.check_call(command, cwd=PROJECT_PATH)


def handle_clean(args: argparse.Namespace) -> None:
    for name, info in get_modules().items():
        if not args.module or name in args.module:
            print("[+] Cleaning project: %s" % name)
            paths = [
                os.path.join(info["path"], "dist"),
                os.path.join(info["path"], "build"),
                os.path.join(info["path"], "src", "%s.egg-info" % info["module"].replace("-", "_")),
            ]
            for path in paths:
                if not os.path.exists(path):
                    print("[-] Path does not exist, skipping: %s" % path)
                elif os.path.isdir(path):
                    print("[-] Removing directory: %s" % path)
                    shutil.rmtree(path)
                else:
                    print("[-] Removing file: %s" % path)
                    os.remove(path)
    if not args.module:
        dist = os.path.join(PROJECT_PATH, "dist")
        if os.path.isdir(dist):
            print("[-] Removing directory: %s" % dist)
            shutil.rmtree(dist)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
