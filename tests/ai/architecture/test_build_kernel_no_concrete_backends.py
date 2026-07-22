#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build-kernel concrete-backend boundary guard.

The Runtime build kernel (``linktools.ai.runtime.builder``) must NOT import any
concrete Storage backend module. The coordinator is injected by the
composition root (the caller that constructs ``RuntimeBuildConfig``); the
build kernel accepts it as an opaque Protocol and never branches on Storage
type. A regression that adds ``from ..storage.filesystem.commit import ...``
or ``from ..storage.sqlalchemy.commit import ...`` to the build kernel would
re-couple it to a concrete backend and must fail this test.

The forbidden module roots:
- ``linktools.ai.storage.filesystem`` -- the file-backed reference backend.
- ``linktools.ai.storage.sqlalchemy`` -- the SQL-backed reference backend.
- ``linktools.ai.storage.artifact_backends`` -- concrete artifact backend
  adapters (the build kernel consumes the abstract ``ArtifactStore`` only).

The composition root (``linktools.ai_cli.runtime``) and tests MAY import
these; the build kernel may not. This makes the boundary a mechanical merge
gate rather than a goodwill convention."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_BUILD_KERNEL = (
    _REPO
    / "linktools-ai"
    / "src"
    / "linktools"
    / "ai"
    / "runtime"
    / "builder.py"
)

# Concrete backend module roots the build kernel must not import. Any import
# whose resolved absolute module starts with one of these is a violation: the
# build kernel has re-coupled to a concrete Storage/Artifact backend.
_FORBIDDEN_MODULE_ROOTS: "tuple[str, ...]" = (
    "linktools.ai.storage.filesystem",
    "linktools.ai.storage.sqlalchemy",
    "linktools.ai.storage.artifact_backends",
)


def _resolved_imports(path: Path) -> "set[str]":
    """Every absolute ``linktools.ai.*`` import in ``path``, with relative
    imports resolved against the file's location.

    Covers both ``import a.b.c`` and ``from .a.b import c`` / ``from ..a.b
    import c`` forms so a forbidden dep cannot slip in via a dotted-relative
    form that the absolute-only check would miss."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    # Each file is at ``linktools/ai/runtime/builder.py``; relative imports
    # resolve against the file's package (``linktools.ai.runtime.builder``).
    file_pkg_parts = ("linktools", "ai", "runtime")
    out: "set[str]" = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("linktools."):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module and node.module.startswith("linktools."):
                    out.add(node.module)
                continue
            # Relative: resolve against the file's package. level=1 is the
            # current package; level=2 is the parent; etc. Drop the last
            # ``level-1`` parts from the file's package to land on the
            # resolved base, then append the dotted module.
            base_parts = list(file_pkg_parts)
            if node.level > 1:
                # level=2 -> parent of file_pkg; level=3 -> grandparent; ...
                base_parts = base_parts[: -(node.level - 1)] or ["linktools"]
            if node.module:
                resolved = ".".join(base_parts) + "." + node.module
            else:
                resolved = ".".join(base_parts)
            if resolved.startswith("linktools."):
                out.add(resolved)
    return out


def test_build_kernel_path_exists() -> None:
    """Sanity: the file under test exists at the expected location."""
    assert _BUILD_KERNEL.is_file(), (
        f"build kernel not found at {_BUILD_KERNEL}; the AST guard cannot run"
    )


def test_build_kernel_imports_no_concrete_storage_backend() -> None:
    """The build kernel must not import any concrete Storage/Artifact backend
    module. The coordinator is injected; the kernel branches on capability
    flags only."""
    imports = _resolved_imports(_BUILD_KERNEL)
    offenders = sorted(
        imp
        for imp in imports
        for root in _FORBIDDEN_MODULE_ROOTS
        if imp == root or imp.startswith(root + ".")
    )
    assert not offenders, (
        "build kernel imports concrete Storage/Artifact backends; the "
        "coordinator must be injected by the composition root, not selected "
        "by the kernel:\n  "
        + "\n  ".join(offenders)
    )


def test_build_kernel_no_storage_type_branching() -> None:
    """The build kernel must not branch on Storage concrete type names. A
    regression that reads ``storage.root`` (a FilesystemStorage-specific
    attribute the abstract ``Storage`` does not declare) or tests
    ``isinstance(storage, FilesystemStorage)`` would defeat the injected
    coordinator contract."""
    text = _BUILD_KERNEL.read_text(encoding="utf-8")
    forbidden_patterns: "tuple[str, ...]" = (
        "isinstance(storage, FilesystemStorage)",
        "isinstance(storage, SqlAlchemyStorage)",
        "isinstance(config.storage, FilesystemStorage)",
        "isinstance(config.storage, SqlAlchemyStorage)",
        'hasattr(storage, "root")',
        'hasattr(config.storage, "root")',
    )
    offenders = [p for p in forbidden_patterns if p in text]
    assert not offenders, (
        "build kernel branches on Storage concrete type or probes for a "
        "FilesystemStorage-specific attribute; the coordinator must be "
                "injected explicitly:\n  " + "\n  ".join(offenders)
    )


def test_runtime_builds_with_injected_coordinator_and_no_concrete_backend(
    tmp_path: Path,
) -> None:
    """A Runtime assembled from the public in-memory external Storage pattern
    + an externally-injected RunCommitCoordinator builds successfully, with
    no concrete reference-backend module loaded by the build kernel.

    This is the functional proof that the composition root contract holds:
    a caller that wires its own coordinator against the public Storage
    Protocol surface can assemble a Runtime without the build kernel ever
    touching ``storage.filesystem`` / ``storage.sqlalchemy`` /
    ``storage.artifact_backends``. The AST guard above proves the static
    side (no forbidden imports); this test proves the dynamic side (a real
    ``build_runtime_components`` call succeeds end-to-end on that
    contract)."""
    import sys

    from linktools.ai.runtime.builder import (
        RuntimeBuildConfig,
        RuntimeSettings,
        build_runtime_components,
    )
    from linktools.ai.runtime.dependencies import RuntimeDependencies
    from linktools.ai.storage.filesystem.commit import (
        FilesystemRunCommitCoordinator,
    )

    # Snapshot the concrete-backend modules already in sys.modules BEFORE
    # the build call so the post-build diff isolates the build kernel's own
    # footprint (modules imported by this test function or earlier test
    # collection are not counted against it).
    pre_build = {
        m
        for m in sys.modules
        if m.startswith(
            (
                "linktools.ai.storage.filesystem",
                "linktools.ai.storage.sqlalchemy",
                "linktools.ai.storage.artifact_backends",
            )
        )
    }

    # The public in-memory external Storage pattern lives in the sibling
    # storage test package; import it through its package path so its
    # internal relative imports resolve.
    from external_adapter import (
        build_in_memory_external_storage,
    )

    storage = build_in_memory_external_storage(root=tmp_path)
    # Composition root: the caller constructs the concrete coordinator from
    # its concrete Storage and injects it. The build kernel never branches
    # on Storage type.
    coordinator = FilesystemRunCommitCoordinator.from_storage(storage)
    config = RuntimeBuildConfig(
        storage=storage,
        providers=RuntimeDependencies(),
        commit_coordinator=coordinator,
        settings=RuntimeSettings(),
    )
    components = build_runtime_components(config)
    # The injected coordinator flows through to the components unchanged.
    assert components.commit_coordinator is coordinator
    # No NEW concrete reference backend module loaded as a side effect of
    # build_runtime_components (the build kernel's job is assembly, not
    # backend selection).
    post_build = {
        m
        for m in sys.modules
        if m.startswith(
            (
                "linktools.ai.storage.filesystem",
                "linktools.ai.storage.sqlalchemy",
                "linktools.ai.storage.artifact_backends",
            )
        )
    }
    new_modules = sorted(post_build - pre_build)
    assert not new_modules, (
        "build_runtime_components loaded concrete backend modules as a side "
        "effect of assembly:\n  " + "\n  ".join(new_modules)
    )


def test_missing_commit_coordinator_fails_fast(tmp_path):
    # The build kernel no longer selects a coordinator from Storage type, so a
    # caller that forgets to inject one must fail fast at build time (not
    # silently degrade to a no-op commit path). Pins the fail-fast contract.
    from linktools.ai.runtime.builder import (
        RuntimeBuildConfig,
        RuntimeSettings,
        build_runtime_components,
    )
    from linktools.ai.runtime.dependencies import RuntimeDependencies
    from linktools.ai.errors import RuntimeInitializationError
    from linktools.ai.storage.facade import FilesystemStorage

    storage = FilesystemStorage(root=tmp_path)
    config = RuntimeBuildConfig(
        storage=storage,
        providers=RuntimeDependencies(),
        commit_coordinator=None,  # forgotten injection
        settings=RuntimeSettings(),
    )
    with pytest.raises(RuntimeInitializationError, match="RunCommitCoordinator must be injected"):
        build_runtime_components(config)

