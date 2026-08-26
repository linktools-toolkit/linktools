#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import typing

import tomlkit
import yaml
from tomlkit.items import Array

MODULE_NAME = "linktools"
TEMPLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))

_REQUIRES_PYTHON_PATTERN = re.compile(r"^>=(\d+)\.(\d+)$")
_GATE_MODULE_PATTERN = re.compile(r"^scripts\.check(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_VERSION_LINE_PATTERN = re.compile(r"(?m)^version:[^\r\n]*$")
_SUPPORTED_CHECKS = {"gate", "ruff", "pytest"}
_GATE_FIELDS = {"modules"}
_RUFF_FIELDS = {"select", "paths"}
_PYTEST_FIELDS = {"paths"}

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


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicated = key in mapping
        except TypeError:
            raise ValueError("mapping keys must be scalar values")
        if duplicated:
            raise ValueError("duplicate YAML key: %s" % key)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


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


def sync_project_template(
    source_path: str,
    target_path: str,
    exist_ok: bool = True,
    **format_vars: typing.Any
) -> None:
    if os.path.exists(target_path) and not exist_ok:
        print("[-] Skipping existing file: %s" % target_path)
        return
    print("[+] Syncing project template: %s -> %s" % (source_path, target_path))
    with open(source_path, "r", encoding="utf-8") as source_file:
        content = source_file.read().format(**format_vars)
    with open(target_path, "w", encoding="utf-8") as target_file:
        target_file.write(content)


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
    check_parser.add_argument(
        "--compatibility",
        action="store_true",
        help="Run Python compatibility gates only",
    )
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


def _selected_modules(
    args: argparse.Namespace,
    modules: "typing.Dict[str, typing.Dict[str, str]]",
) -> "typing.Tuple[str, ...]":
    if args.module:
        return tuple(dict.fromkeys(args.module))
    return tuple(modules.keys())


def _read_requires_python(project: str, project_path: str) -> "typing.Tuple[int, int]":
    path = os.path.join(project_path, "pyproject.toml")
    with open(path, "r", encoding="utf-8") as file:
        data = tomlkit.parse(file.read())
    value = data.get("project", {}).get("requires-python")
    match = _REQUIRES_PYTHON_PATTERN.match(value) if isinstance(value, str) else None
    if not match:
        print(
            "[-] %s requires-python must use the exact >=X.Y form, got %r" % (project, value),
            file=sys.stderr,
        )
        raise SystemExit(1)
    return int(match.group(1)), int(match.group(2))


def _require_mapping(value: typing.Any, label: str) -> dict:
    if not isinstance(value, dict):
        print("[-] %s must be a mapping" % label, file=sys.stderr)
        raise SystemExit(1)
    return value


def _unknown_fields(value: dict, allowed: "typing.Set[str]", label: str) -> "typing.Tuple[str, ...]":
    if not all(isinstance(key, str) for key in value):
        print("[-] %s field names must be strings" % label, file=sys.stderr)
        raise SystemExit(1)
    return tuple(sorted(set(value) - allowed))


def _require_string_list(value: typing.Any, label: str) -> "typing.Tuple[str, ...]":
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        print("[-] %s must be a non-empty list of strings" % label, file=sys.stderr)
        raise SystemExit(1)
    return tuple(value)


def _resolve_paths(
    project: str,
    project_path: str,
    values: "typing.Tuple[str, ...]",
    label: str,
) -> "typing.Tuple[str, ...]":
    resolved = []
    for value in values:
        if os.path.isabs(value):
            print("[-] %s %s path must be relative: %s" % (project, label, value), file=sys.stderr)
            raise SystemExit(1)
        path = os.path.realpath(os.path.join(project_path, value))
        try:
            inside_repository = os.path.commonpath((PROJECT_PATH, path)) == PROJECT_PATH
        except ValueError:
            inside_repository = False
        if not inside_repository:
            print("[-] %s %s path escapes repository: %s" % (project, label, value), file=sys.stderr)
            raise SystemExit(1)
        if not os.path.exists(path):
            print("[-] %s %s path does not exist: %s" % (project, label, value), file=sys.stderr)
            raise SystemExit(1)
        resolved.append(path)
    return tuple(resolved)


def _resolve_gate_modules(project: str, values: "typing.Tuple[str, ...]") -> "typing.Tuple[str, ...]":
    for module in values:
        if not _GATE_MODULE_PATTERN.match(module):
            print(
                "[-] %s gate module must be under scripts.check: %s" % (project, module),
                file=sys.stderr,
            )
            raise SystemExit(1)
        relative = module.replace(".", os.sep)
        package_main = os.path.join(PROJECT_PATH, relative, "__main__.py")
        module_file = os.path.join(PROJECT_PATH, relative + ".py")
        if not os.path.isfile(package_main) and not os.path.isfile(module_file):
            print(
                "[-] %s gate module is not executable with python -m: %s" % (project, module),
                file=sys.stderr,
            )
            raise SystemExit(1)
    return values


def _load_release_config(project: str, project_path: str) -> "typing.Dict[str, typing.Any]":
    path = os.path.join(project_path, "release.yml")
    if not os.path.isfile(path):
        print("[-] %s release config is missing: %s" % (project, path), file=sys.stderr)
        raise SystemExit(1)
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = yaml.load(file, Loader=_UniqueKeyLoader)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print("[-] %s release config is invalid: %s" % (project, error), file=sys.stderr)
        raise SystemExit(1)

    root = _require_mapping(data, "%s release config" % project)
    unknown_root = _unknown_fields(root, {"checks"}, "%s release config" % project)
    if unknown_root:
        print("[-] %s release config has unknown field(s): %s" % (project, ", ".join(unknown_root)), file=sys.stderr)
        raise SystemExit(1)
    checks = _require_mapping(root.get("checks"), "%s checks" % project)
    unknown_checks = _unknown_fields(checks, _SUPPORTED_CHECKS, "%s checks" % project)
    if unknown_checks:
        print("[-] %s has unknown check(s): %s" % (project, ", ".join(unknown_checks)), file=sys.stderr)
        raise SystemExit(1)

    result = {}
    if "gate" in checks:
        gate = _require_mapping(checks["gate"], "%s checks.gate" % project)
        unknown = _unknown_fields(gate, _GATE_FIELDS, "%s checks.gate" % project)
        if unknown:
            print("[-] %s checks.gate has unknown field(s): %s" % (project, ", ".join(unknown)), file=sys.stderr)
            raise SystemExit(1)
        modules = _require_string_list(gate.get("modules"), "%s checks.gate.modules" % project)
        result["gate"] = {
            "modules": _resolve_gate_modules(project, modules),
        }

    if "ruff" in checks:
        ruff = _require_mapping(checks["ruff"], "%s checks.ruff" % project)
        unknown = _unknown_fields(ruff, _RUFF_FIELDS, "%s checks.ruff" % project)
        if unknown:
            print("[-] %s checks.ruff has unknown field(s): %s" % (project, ", ".join(unknown)), file=sys.stderr)
            raise SystemExit(1)
        select = _require_string_list(ruff.get("select"), "%s checks.ruff.select" % project)
        paths = _require_string_list(ruff.get("paths"), "%s checks.ruff.paths" % project)
        result["ruff"] = {
            "select": select,
            "paths": _resolve_paths(project, project_path, paths, "ruff"),
        }

    if "pytest" in checks:
        pytest = _require_mapping(checks["pytest"], "%s checks.pytest" % project)
        unknown = _unknown_fields(pytest, _PYTEST_FIELDS, "%s checks.pytest" % project)
        if unknown:
            print("[-] %s checks.pytest has unknown field(s): %s" % (project, ", ".join(unknown)), file=sys.stderr)
            raise SystemExit(1)
        paths = _require_string_list(pytest.get("paths"), "%s checks.pytest.paths" % project)
        result["pytest"] = {
            "paths": _resolve_paths(project, project_path, paths, "pytest"),
        }
    return result


def _compatible_projects(
    args: argparse.Namespace,
    projects: "typing.Tuple[str, ...]",
    requirements: "typing.Dict[str, typing.Tuple[int, int]]",
) -> "typing.Tuple[str, ...]":
    current = sys.version_info[:2]
    incompatible = [project for project in projects if current < requirements[project]]
    if args.module and incompatible:
        required = max(requirements[project] for project in incompatible)
        owners = [project for project in incompatible if requirements[project] == required]
        print(
            "[-] Python %d.%d is required by %s; current interpreter is %d.%d"
            % (required[0], required[1], ", ".join(owners), current[0], current[1]),
            file=sys.stderr,
        )
        raise SystemExit(1)
    for project in incompatible:
        required = requirements[project]
        print(
            "[+] Skipping %s: requires Python >=%d.%d, current interpreter is %d.%d"
            % (project, required[0], required[1], current[0], current[1])
        )
    return tuple(project for project in projects if project not in incompatible)


def _check_environment() -> "typing.Dict[str, str]":
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _run_check(command: "typing.Sequence[str]", environment: "typing.Dict[str, str]") -> None:
    print("[+] Running: %s" % " ".join(command))
    subprocess.check_call(list(command), cwd=PROJECT_PATH, env=environment)


def _run_python36_gate(
    projects: "typing.Tuple[str, ...]",
    requirements: "typing.Dict[str, typing.Tuple[int, int]]",
    modules: "typing.Dict[str, typing.Dict[str, str]]",
    environment: "typing.Dict[str, str]",
) -> None:
    compatible = [project for project in projects if requirements[project] <= (3, 6)]
    if not compatible:
        return
    scanner = os.path.join(PROJECT_PATH, "scripts", "check", "python36.py")
    paths = [os.path.join(PROJECT_PATH, "manage.py"), scanner]
    paths.extend(os.path.join(modules[project]["path"], "src") for project in compatible)
    print("[+] Python 3.6 compatibility: %s" % ", ".join(compatible))
    _run_check([sys.executable, scanner] + paths, environment)


def _run_gate(project: str, check: "typing.Dict[str, typing.Any]", environment: "typing.Dict[str, str]") -> None:
    for module in check["modules"]:
        print("[+] %s: gate %s" % (project, module))
        _run_check([sys.executable, "-m", module], environment)


def _run_ruff(project: str, check: "typing.Dict[str, typing.Any]", environment: "typing.Dict[str, str]") -> None:
    ruff = shutil.which("ruff")
    if not ruff:
        print("[-] ruff is required; install repository requirements first", file=sys.stderr)
        raise SystemExit(1)
    print("[+] %s: ruff" % project)
    command = [ruff, "check", "--no-cache", "--select", ",".join(check["select"])]
    command.extend(check["paths"])
    _run_check(command, environment)


def _run_pytest(project: str, check: "typing.Dict[str, typing.Any]", environment: "typing.Dict[str, str]") -> None:
    print("[+] %s: pytest" % project)
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    command.extend(check["paths"])
    _run_check(command, environment)


def _normalize_version(value: str) -> str:
    version = value[1:] if value.startswith("v") else value
    if not version:
        raise ValueError("VERSION must not be empty")
    return version


def _replace_linktools_version(path: str, version: str) -> None:
    with open(path, "r", encoding="utf-8") as file:
        content = file.read()
    try:
        data = yaml.load(content, Loader=_UniqueKeyLoader)
    except (ValueError, yaml.YAMLError) as error:
        raise ValueError("invalid Linktools config %s: %s" % (path, error))
    if not isinstance(data, dict):
        raise ValueError("Linktools config must be a mapping: %s" % path)
    current = data.get("version")
    if not isinstance(current, str) or not current:
        raise ValueError("Linktools config version is missing: %s" % path)
    matches = list(_VERSION_LINE_PATTERN.finditer(content))
    if len(matches) != 1:
        raise ValueError("Linktools config must contain exactly one top-level version line: %s" % path)
    replacement = "version: %s" % json.dumps(version)
    updated = content[:matches[0].start()] + replacement + content[matches[0].end():]
    with open(path, "w", encoding="utf-8") as file:
        file.write(updated)


def _restore_files(backups: "typing.Dict[str, str]") -> None:
    for path, content in backups.items():
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)


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
            sync_project_template(
                os.path.join(TEMPLATE_PATH, "linktools.yml"),
                os.path.join(info["path"], "linktools.yml"),
                exist_ok=False,
                **info
            )
            sync_project_file(
                os.path.join(TEMPLATE_PATH, "capability.jinja2"),
                os.path.join(info["path"], "capability.jinja2"),
                exist_ok=True,
            )
            sync_project_file(
                os.path.join(TEMPLATE_PATH, "release.yml"),
                os.path.join(info["path"], "release.yml"),
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
    modules = get_modules()
    projects = _selected_modules(args, modules)
    requirements = {}
    checks = {}
    for project in projects:
        project_path = modules[project]["path"]
        requirements[project] = _read_requires_python(project, project_path)
        checks[project] = _load_release_config(project, project_path)

    compatible = _compatible_projects(args, projects, requirements)
    environment = _check_environment()
    _run_python36_gate(compatible, requirements, modules, environment)

    if not args.compatibility:
        for project in compatible:
            project_checks = checks[project]
            if "gate" in project_checks:
                _run_gate(project, project_checks["gate"], environment)
            if "ruff" in project_checks:
                _run_ruff(project, project_checks["ruff"], environment)
            if "pytest" in project_checks:
                _run_pytest(project, project_checks["pytest"], environment)

    mode = "compatibility" if args.compatibility else "check"
    print("[+] %s passed: %s" % (mode.capitalize(), ", ".join(compatible)))


def handle_build(args: argparse.Namespace) -> None:
    modules = get_modules()
    projects = _selected_modules(args, modules)
    version_value = os.environ.get("VERSION")
    backups = {}
    try:
        if version_value is not None:
            version = _normalize_version(version_value)
            for project in projects:
                path = os.path.join(modules[project]["path"], "linktools.yml")
                with open(path, "r", encoding="utf-8") as file:
                    backups[path] = file.read()
            for project in projects:
                path = os.path.join(modules[project]["path"], "linktools.yml")
                print("[+] Setting version for project: %s to %s" % (project, version))
                _replace_linktools_version(path, version)

        for project in projects:
            info = modules[project]
            print("[+] Building project: %s, path: %s" % (project, info["path"]))
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
    except BaseException:
        if backups:
            _restore_files(backups)
        raise


def handle_verify(args: argparse.Namespace) -> None:
    projects = _selected_modules(args, get_modules())
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
