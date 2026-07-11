#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only repository status/validation summary for ``repo status``/
``repo validate``: manifest presence, project/cntr-component name/version,
compatibility result, manifest hash, Git revision and dirty state. Never
imports the repository's ``container.py``.

Only the ``cntr`` component's own fields are ever surfaced in detail --
other components (e.g. a future ``ai``) are named (so a project's full set
of declared capabilities is visible) but never dumped in full, since their
config/metadata may hold something this command has no business printing.
"""
import hashlib
import os
from typing import TYPE_CHECKING

from .manifest import ContainerManifestError

if TYPE_CHECKING:
    from typing import Any
    from ..manager import ContainerManager


def describe_repository(
        manager: "ContainerManager", url: str, meta: "dict[str, Any]", check_runtime: bool = False,
) -> "dict[str, Any]":
    repo_path = meta.get("repo_path")
    info: "dict[str, Any]" = dict(url=url, type=meta.get("type", "unknown"), repo_path=repo_path)

    manifest_path = os.path.join(repo_path, manager.manifest_policy.loader.file_name) if repo_path else None
    if not manifest_path or not os.path.exists(manifest_path):
        info["manifest"] = "legacy"
        return info

    info["manifest"] = "present"
    try:
        with open(manifest_path, "rb") as f:
            info["manifest_sha256"] = hashlib.sha256(f.read()).hexdigest()
        manifest, component = manager.manifest_policy.load_and_get_component(repo_path)
    except OSError as exc:
        message = "Unable to read %s: %s" % (manager.manifest_policy.loader.file_name, exc.strerror or exc)
    except ContainerManifestError as exc:
        message = str(exc)
    else:
        message = None

    if message is not None:
        info["manifest_error"] = message
        info["compatible"] = False
        return info

    info["kind"] = manifest.kind
    info["project_name"] = manifest.name
    info["project_version"] = manifest.version
    info["project_requires"] = dict(manifest.requires)
    info["components"] = sorted(manifest.components.keys())
    info["cntr_component_schema_version"] = component.schema_version
    info["cntr_requires"] = dict(component.requires)

    issues = list(manager.manifest_policy.check_host_requirements(manifest))
    if check_runtime:
        issues += manager.manifest_policy.check_runtime_requirements(manifest)

    info["compatible"] = not issues
    if issues:
        info["compatibility_issues"] = [issue.message for issue in issues]

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
