#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only repository status/validation summary for ``repo status``/
``repo validate``: local file config presence, declared
``requires.linktools-cntr``, compatibility result, file hash, Git revision
and dirty state. Never imports the repository's ``container.py``.
"""
import hashlib
import os
from typing import TYPE_CHECKING

from linktools.core import ensure_requirement
from linktools.errors import ConfigError, ConfigValidationError

from ...capabilities.cntr import __cap_cntr__

if TYPE_CHECKING:
    from typing import Any
    from ..manager import ContainerManager


def describe_repository(
        manager: "ContainerManager", url: str, meta: "dict[str, Any]",
) -> "dict[str, Any]":
    repo_path = meta.get("repo_path")
    info: "dict[str, Any]" = dict(url=url, type=meta.get("type", "unknown"), repo_path=repo_path)

    from linktools.core import LinktoolsFileConfigLoader
    local_path = os.path.join(repo_path, LinktoolsFileConfigLoader.local_file_name) if repo_path else None
    if not local_path or not os.path.exists(local_path):
        info["local_config"] = "absent"
        info["compatible"] = True
        return info

    info["local_config"] = "present"
    try:
        with open(local_path, "rb") as f:
            info["local_config_sha256"] = hashlib.sha256(f.read()).hexdigest()
        file_config = manager.environ.load_file_config(local_root=repo_path)
    except OSError as exc:
        info["local_config_error"] = "Unable to read %s: %s" % (local_path, exc.strerror or exc)
        info["compatible"] = False
        return info
    except ConfigError as exc:
        info["local_config_error"] = str(exc)
        info["compatible"] = False
        return info

    info["requires"] = dict(file_config.local_config.requires)

    try:
        ensure_requirement(file_config.local_config, "linktools-cntr", __cap_cntr__.version)
        info["compatible"] = True
    except ConfigValidationError as exc:
        info["compatible"] = False
        info["compatibility_issues"] = [str(exc)]

    if info["type"] == "git" and repo_path and os.path.exists(repo_path):
        try:
            from linktools.git import GitRepository
            from dulwich.errors import NotGitRepository
            repo = GitRepository(manager.environ, repo_path)
            info["revision"] = repo.head_sha()
            info["dirty"] = repo.is_dirty()
        except NotGitRepository:
            pass
        except Exception:  # noqa: BLE001 - status must stay read-only & non-fatal
            pass

    return info
