#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flare generated configuration and application."""
import os
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ..container import ContainerError
from ..integration import ExposeLink

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from ..artifacts import GeneratedCandidate
    from ..container import BaseContainer
    from ..context import EventContext


class FlareGeneration:
    """Own the builtin flare consumer without extending container hooks."""

    def __init__(self, container: "BaseContainer") -> None:
        self.container = container

    def prepare(self, context: "EventContext") -> None:
        pass

    def _iter_links(self) -> "Iterator[ExposeLink]":
        manager = self.container.manager
        snapshot = manager.integration_snapshot
        if "flare" not in snapshot:
            return
        producers = sorted(
            (name for name, consumers in snapshot.items() if consumers),
            key=lambda name: manager.containers[name].order,
        )
        for name in producers:
            consumers = snapshot[name]
            for local_id in consumers.get("nginx", {}):
                expose = manager.nginx_sites[(name, local_id)].expose
                if expose is not None:
                    yield expose
            declarations = consumers.get("flare", ())
            for expose in declarations.values() if isinstance(declarations, Mapping) else declarations:
                yield expose

    def render(self, generation_id: str) -> "dict[str, str]":

        categories = OrderedDict()
        apps = []

        for expose in self._iter_links():
            if not isinstance(expose, ExposeLink) or not expose.is_valid:
                continue
            category = expose.category
            existing = categories.get(category.name)
            if existing is None:
                existing = (category.desc, [])
                categories[category.name] = existing
            elif existing[0] != category.desc:
                raise ContainerError(
                    f"Conflicting description for category {category.name!r}")
            existing[1].append(expose)
            if category.name == "public":
                apps.append(expose)

        data = {"links": []}
        for app in apps:
            data["links"].append({
                "name": app.name,
                "desc": app.desc,
                "icon": app.icon,
                "link": app.url,
            })
        result = {"apps.yml": yaml.safe_dump(data, allow_unicode=True)}

        data = {"categories": [], "links": []}
        names = [name for name in ("private", "container", "other") if name in categories]
        names.extend(name for name in categories if name not in ("private", "container", "other"))
        for name in names:
            description, links = categories[name]
            if name == "public":
                continue
            if not links:
                continue
            data["categories"].append({
                "id": name,
                "title": description,
            })
            for link in links:
                data["links"].append({
                    "category": name,
                    "name": link.name,
                    "icon": link.icon,
                    "link": link.url,
                })
        result["bookmarks.yml"] = yaml.safe_dump(data, allow_unicode=True)
        return result

    def validate(self, candidate: "GeneratedCandidate", context: "EventContext") -> None:
        group = self.container.get_config("DOCKER_GID", type=int)
        for name in ("apps.yml", "bookmarks.yml"):
            path = Path(candidate.path) / name
            yaml.safe_load(path.read_text())
            # The service's configured group needs read access; the host owner
            # retains access for content comparison and future rollback.
            if path.stat().st_gid != group:
                self.container.runtime.create_process("chgrp", str(group), str(path), privilege=True).check_call()
            path.chmod(0o640)

    def apply(self, candidate: "GeneratedCandidate", context: "EventContext",
              services: "Iterable[str]") -> None:
        if "flare" not in services:
            return
        app = self.container.get_app_path("app")
        app.mkdir(parents=True, exist_ok=True)
        migrated = []
        try:
            for name in ("apps.yml", "bookmarks.yml"):
                path = app / name
                target = "../generated/current/" + name
                if path.is_symlink() and os.readlink(str(path)) == target:
                    continue
                backup = None
                if path.exists() or path.is_symlink():
                    backup = app / (name + ".pre-cntr")
                    if backup.exists() or backup.is_symlink():
                        raise ContainerError("Flare migration backup already exists for " + name)
                    path.rename(backup)
                migrated.append((path, backup))
                temporary = app / (name + ".cntr-link")
                if temporary.exists() or temporary.is_symlink():
                    temporary.unlink()
                temporary.symlink_to(target)
                os.replace(str(temporary), str(path))
            runner = self.container.manager.compose_runner
            recreate = candidate.changed or not runner.is_generation_current(context, "flare", candidate)
            runner.apply_service(context, "flare", recreate=recreate)
            runner.wait_service_running(context, "flare")
        except Exception:
            for path, backup in reversed(migrated):
                if path.is_symlink():
                    path.unlink()
                if backup is not None:
                    backup.rename(path)
            raise
