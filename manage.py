#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import shutil
import subprocess
import sys

import tomlkit
from tomlkit.items import Array

MODULE_NAME = "linktools"
TEMPLATE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))


def get_modules():
    modules = {}
    for name in os.listdir(PROJECT_PATH):
        path = os.path.join(PROJECT_PATH, name)
        if os.path.isdir(path):
            if name == MODULE_NAME:
                modules[MODULE_NAME] = dict(
                    name="",
                    module=MODULE_NAME,
                    path=path,
                )
            elif name.startswith(MODULE_NAME):
                match = re.match(rf"{MODULE_NAME}[-_](.+)", name)
                if match:
                    simple_name = match.group(1)
                    module_name = f"{MODULE_NAME}-{simple_name}"
                    modules[module_name] = dict(
                        name=simple_name,
                        module=module_name,
                        path=path,
                    )
    return modules


def update_toml_recursive(source_data, target_data, **format_kwargs):
    """
    递归更新 TOML 数据，支持字典、列表及字符串格式化。
    """
    # 处理字典/表结构
    if isinstance(source_data, dict):
        for key, value in source_data.items():
            if isinstance(value, (dict, list)):
                # 确保目标路径存在
                if key not in target_data:
                    target_data[key] = tomlkit.table() if isinstance(value, dict) else tomlkit.array()
                update_toml_recursive(value, target_data[key], **format_kwargs)
            elif isinstance(value, str):
                target_data[key] = value.format(**format_kwargs)
            else:
                target_data[key] = value

    # 处理列表/数组结构
    elif isinstance(source_data, list):
        # 数组处理策略：通常数组作为整体更新，
        # 但为了处理内部字符串，我们需要遍历它
        new_array = tomlkit.array()
        for item in source_data:
            if isinstance(item, str):
                new_array.append(item.format(**format_kwargs))
            elif isinstance(item, (dict, list)):
                # 递归处理数组嵌套的字典或数组
                # 创建一个临时容器进行递归，然后存入新数组
                temp_container = tomlkit.inline_table() if isinstance(item, dict) else tomlkit.array()
                update_toml_recursive(item, temp_container, **format_kwargs)
                new_array.append(temp_container)
            else:
                new_array.append(item)

        # 将处理完的新数组赋值回目标（如果是 tomlkit 对象则更新内容）
        if isinstance(target_data, Array):
            target_data.clear()
            for val in new_array:
                target_data.append(val)


def sync_pyproject_toml(source_path, target_path, **format_vars):
    print(f"[+] Syncing pyproject.toml: {source_path} -> {target_path}")

    # 读取源文件
    with open(source_path, "r", encoding="utf-8") as f:
        source_doc = tomlkit.parse(f.read())

    # 读取目标文件（如果不存在则创建新文档）
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            target_doc = tomlkit.parse(f.read())
    except FileNotFoundError:
        target_doc = tomlkit.document()

    # 执行递归更新
    update_toml_recursive(source_doc, target_doc, **format_vars)

    # 写回目标文件
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(target_doc))


def sync_project_file(source_path, target_path, exist_ok=True):
    if not os.path.exists(target_path) or exist_ok:
        print(f"[+] Syncing project file: {source_path} -> {target_path}")
        shutil.copy2(source_path, target_path)
    else:
        print(f"[-] Skipping existing file: {target_path}")


# ======================
# argparse
# ======================

def build_parser() -> argparse.ArgumentParser:
    modules = get_modules()

    parser = argparse.ArgumentParser(
        description="Project management tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # -------- init 子命令 --------
    init_parser = subparsers.add_parser(
        "init",
        help="Initialize a new project",
    )
    init_parser.add_argument(
        "module",
        choices=sorted(modules.keys()),
        nargs="*",
        help="Module to init",
    )
    init_parser.set_defaults(func=handle_init)

    # -------- install 子命令 --------
    install_parser = subparsers.add_parser(
        "install",
        help="Install project modules",
    )
    install_parser.add_argument(
        "module",
        choices=sorted(modules.keys()),
        nargs="*",
        help="Module to install",
    )
    install_parser.add_argument(
        "-e",
        "--editable",
        action="store_true",
        help="Install in editable mode",
    )
    install_parser.add_argument(
        "--no-isolation",
        action="store_true",
        help="Disable build isolation (pip compatible)",
    )
    install_parser.set_defaults(func=handle_install)

    # -------- build 子命令 --------
    build_parser = subparsers.add_parser(
        "build",
        help="Build project modules",
    )
    build_parser.add_argument(
        "module",
        choices=sorted(modules.keys()),
        nargs="*",
        help="Module to build",
    )
    build_parser.set_defaults(func=handle_build)

    # -------- clean 子命令 --------
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean project modules files",
    )
    clean_parser.add_argument(
        "module",
        choices=sorted(modules.keys()),
        nargs="*",
        help="Module to clean",
    )
    clean_parser.set_defaults(func=handle_clean)

    return parser


def handle_init(args: argparse.Namespace):
    for name, info in get_modules().items():
        if name == MODULE_NAME:
            continue
        if not args.module or name in args.module:
            print(f"[+] Initializing project: {name}")
            sync_pyproject_toml(
                os.path.join(TEMPLATE_PATH, "pyproject.template"),
                os.path.join(info.get("path"), "pyproject.toml"),
                **info,
            )
            sync_project_file(
                os.path.join(TEMPLATE_PATH, "capability.jinja2"),
                os.path.join(info.get("path"), "capability.jinja2"),
                exist_ok=True,
            )
            sync_project_file(
                os.path.join(TEMPLATE_PATH, "MANIFEST.in"),
                os.path.join(info.get("path"), "MANIFEST.in"),
                exist_ok=True,
            )
            sync_project_file(
                os.path.join(TEMPLATE_PATH, "requirements.yml"),
                os.path.join(info.get("path"), "requirements.yml"),
                exist_ok=False,
            )


def handle_install(args: argparse.Namespace):
    paths = []
    for name, info in get_modules().items():
        if not args.module or name in args.module:
            print(f"[+] Adding module to install list: {name}")
            paths.append(info.get("path"))
    pip_args = [sys.executable, "-m", "pip", "install"]
    for path in paths:
        if args.editable:
            pip_args.append("-e")
        pip_args.append(path)
    if args.no_isolation:
        pip_args.append("--no-build-isolation")

    print(f"[+] Running pip install with arguments: {' '.join(pip_args)}")
    subprocess.check_call(pip_args)


def handle_build(args: argparse.Namespace):
    version = os.environ.get("VERSION", None)
    for name, info in get_modules().items():
        if not args.module or name in args.module:
            print(f"[+] Building project: {name}, path: {info.get('path')}")
            subprocess.check_call([
                sys.executable, "-m", "build",
                "--sdist", "--wheel",
                "--outdir", os.path.join(PROJECT_PATH, "dist"),
                info.get("path"),
            ], cwd=PROJECT_PATH)
            if version is not None:
                print(f"[+] Setting version for project: {name} to {version}")
                with open(os.path.join(info.get("path"), ".version"), "wt", encoding="utf-8") as fd:
                    fd.write(version)


def handle_clean(args: argparse.Namespace):
    for name, info in get_modules().items():
        if not args.module or name in args.module:
            print(f"[+] Cleaning project: {name}")
            paths = []
            paths.append(os.path.join(info.get("path"), "dist"))
            paths.append(os.path.join(info.get("path"), "build"))
            paths.append(os.path.join(info.get("path"), "src", f"{info.get('module').replace('-', '_')}.egg-info"))
            for path in paths:
                if os.path.exists(path):
                    if os.path.isdir(path):
                        print(f"[-] Removing directory: {path}")
                        shutil.rmtree(path)
                    else:
                        print(f"[-] Removing file: {path}")
                        os.remove(path)
                else:
                    print(f"[-] Path does not exist, skipping: {path}")


# ======================
# main
# ======================

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
