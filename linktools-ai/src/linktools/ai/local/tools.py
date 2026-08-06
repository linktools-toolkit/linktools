#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Trusted local-coding tools rooted at one project directory."""

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from linktools.core import environ

logger = environ.get_logger("ai.local.tools")

if TYPE_CHECKING:
    from collections.abc import Callable


def build_local_tools(root: "str | Path") -> "tuple[Callable[..., Any], ...]":
    """Build the local file and shell functions exposed to an Agent."""
    project_root = Path(root).expanduser().resolve()

    async def list_dir(path: str = ".", recursive: bool = False) -> "dict[str, Any]":
        """List files below the project root."""
        try:
            target = _resolve_path(project_root, path)
        except ValueError as error:
            return {"error": str(error)}
        if not target.is_dir():
            return {"error": f"not a directory: {path}"}
        iterator = target.rglob("*") if recursive else target.iterdir()
        entries = []
        for item in sorted(iterator, key=lambda value: value.relative_to(target).as_posix()):
            entries.append({
                "path": item.relative_to(project_root).as_posix(),
                "type": "directory" if item.is_dir() else "file",
            })
            if len(entries) >= 500:
                break
        logger.info("local tool completed name=list_dir path=%s count=%s", path, len(entries))
        return {"path": target.relative_to(project_root).as_posix(), "entries": entries}

    async def read_file(path: str, max_chars: int = 6000) -> "dict[str, Any]":
        """Read a UTF-8 text file below the project root."""
        try:
            target = _resolve_path(project_root, path)
        except ValueError as error:
            return {"error": str(error)}
        if not target.is_file():
            return {"error": f"not a file: {path}"}
        content = await asyncio.to_thread(target.read_text, encoding="utf-8")
        limit = max(1, int(max_chars))
        logger.info("local tool completed name=read_file path=%s", path)
        return {"path": target.relative_to(project_root).as_posix(), "content": content[:limit], "truncated": len(content) > limit}

    async def write_file(path: str, content: Any) -> "dict[str, Any]":
        """Write UTF-8 text below the project root."""
        try:
            target = _resolve_path(project_root, path)
        except ValueError as error:
            return {"error": str(error)}
        target.parent.mkdir(parents=True, exist_ok=True)
        value = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, indent=2)
        await asyncio.to_thread(target.write_text, value, encoding="utf-8")
        logger.info("local tool completed name=write_file path=%s bytes=%s", path, len(value.encode("utf-8")))
        return {"path": target.relative_to(project_root).as_posix(), "bytes": len(value.encode("utf-8"))}

    async def bash(command: str, timeout_ms: "int | None" = None) -> "dict[str, Any]":
        """Run a shell command with the project root as its working directory."""
        if not command.strip():
            return {"error": "command must not be empty"}
        timeout = max(1, int(timeout_ms or 60_000)) / 1000
        logger.info("local tool started name=bash command=%s", command)
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(project_root),
            env={key: value for key, value in os.environ.items() if key in {"PATH", "HOME", "LANG", "LC_ALL"}},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return {"exit_code": None, "error": f"timeout after {timeout_ms or 60_000}ms"}
        logger.info("local tool completed name=bash exit_code=%s", process.returncode)
        return {
            "exit_code": process.returncode,
            "stdout": stdout.decode("utf-8", errors="replace")[-16_000:],
            "stderr": stderr.decode("utf-8", errors="replace")[-4_000:],
        }

    return (list_dir, read_file, write_file, bash)


def _resolve_path(root: Path, path: str) -> Path:
    target = (root / path).expanduser().resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"path escapes project root: {path}") from error
    return target


__all__ = ["build_local_tools"]
