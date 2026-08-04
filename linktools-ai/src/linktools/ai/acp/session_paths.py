#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP session working-directory validation."""

import os
from pathlib import Path

from .errors import request_error


def validate_session_paths(
    *, project_root: "str | Path", cwd: str, additional_directories: "list[str] | None"
) -> "tuple[str, tuple[str, ...]]":
    project = Path(os.path.normcase(str(Path(project_root).resolve(strict=True))))
    target = Path(os.path.normcase(str(Path(cwd).resolve(strict=True))))
    if not target.is_dir() or not _contained(target, project):
        raise request_error("invalid_cwd")
    additional = []
    for value in additional_directories or ():
        path = Path(os.path.normcase(str(Path(value).resolve(strict=True))))
        if not path.is_dir():
            raise request_error("invalid_additional_directory")
        additional.append(str(path))
    return str(target), tuple(sorted(set(additional)))


def _contained(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = ["validate_session_paths"]
