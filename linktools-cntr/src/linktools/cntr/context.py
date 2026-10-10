#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Mapping
    from .container import BaseContainer
    from .runtime.inspect import ProjectRuntimeState


@dataclass
class OperationContext:
    """Operation targets, prepared file inputs and hook extension data."""

    actions: "list[str] | None" = None
    project_containers: "list[BaseContainer] | None" = None
    target_containers: "list[BaseContainer] | None" = None
    is_full_project: bool = True
    metadata: "dict[str, Any]" = field(default_factory=dict)
    target_services: "tuple[str, ...] | None" = None
    initial_runtime_state: "ProjectRuntimeState | None" = None
    compose_model: "dict[str, Any] | None" = None
    prepared_dirs: "dict[str, Path]" = field(default_factory=dict)
    refresh_services: "frozenset[str]" = frozenset()

    def write_files(self, container: "BaseContainer", files: "Mapping[str, str]", *,
                    mode: int = 0o600, group: "int | None" = None) -> "Path":
        """Prepare one immutable file tree without changing running services."""
        from .artifacts import stage_files
        if container.name in self.prepared_dirs:
            raise ValueError("Files already prepared for " + container.name)
        path = stage_files(container, files, mode=mode, group=group)
        self.prepared_dirs[container.name] = path
        return path

    def file_path(self, container: "BaseContainer", *parts: str) -> "Path":
        """Locate this operation's prepared files for a native check."""
        from pathlib import PurePosixPath
        path = PurePosixPath(*parts)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Prepared file path must stay within its tree")
        return self.prepared_dirs[container.name].joinpath(*parts)
