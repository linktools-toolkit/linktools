#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cntr's own policy for the generic ``.linktools.json`` project manifest
(``linktools.core.ManifestLoader``): requiring a ``components.cntr`` block,
enforcing its ``schema_version``, and registering the Docker/Compose
``runtime:*`` requirement resolvers cntr alone understands.

Security boundary and JSON-only parsing are handled by
``linktools.core.ManifestLoader`` -- this module only adds cntr-specific
policy on top of an already-validated generic manifest.
"""
from typing import TYPE_CHECKING

from linktools.core import ManifestLoader, RequirementResolverRegistry, RequirementStatus
from linktools.errors import ManifestError, ManifestSchemaUnsupported

from ..container import ContainerError

if TYPE_CHECKING:
    from typing import Optional
    from linktools.core import LinktoolsManifest, ManifestComponent, RequirementResult
    from linktools.types import PathType
    from ..manager import ContainerManager


class ContainerManifestError(ContainerError):
    pass


class ContainerManifestInvalid(ContainerManifestError):
    pass


class ContainerManifestSchemaUnsupported(ContainerManifestInvalid):
    pass


class ContainerIncompatible(ContainerManifestError):
    pass


class ContainerManifestPolicy:
    """Load, validate and compare a repository's ``.linktools.json``
    from cntr's point of view: only the ``components.cntr`` block matters
    here -- other components (e.g. ``ai``) are present in the file but
    entirely ignored by this policy."""

    component_name = "cntr"
    supported_component_versions = (1,)

    def __init__(self, manager: "ContainerManager"):
        self.manager = manager
        self.loader = ManifestLoader()
        self.registry = RequirementResolverRegistry.default()
        self.registry.register_exact("runtime:docker-engine", self._resolve_docker_engine, phase="runtime",
                                     display_name="docker-engine")
        self.registry.register_exact("runtime:docker-compose", self._resolve_docker_compose, phase="runtime",
                                     display_name="docker-compose")

    def load(self, repo_path: "PathType") -> "Optional[LinktoolsManifest]":
        """``None`` means a project without a manifest file -- that is
        not an error and must not warn. Any other load failure (bad JSON,
        wrong kind, unsupported top-level schema_version, ...) is raised as
        ``ContainerManifestInvalid``/``ContainerManifestSchemaUnsupported``
        so cntr callers only ever need to catch ``ContainerManifestError``."""
        try:
            return self.loader.load(repo_path)
        except ManifestSchemaUnsupported as exc:
            raise ContainerManifestSchemaUnsupported(str(exc)) from exc
        except ManifestError as exc:
            raise ContainerManifestInvalid(str(exc)) from exc

    def get_component(self, manifest: "LinktoolsManifest") -> "ManifestComponent":
        """Raise unless ``manifest`` declares a ``components.cntr`` block
        this cntr version supports. A generic manifest with no ``cntr``
        component simply hasn't opted this project into cntr at all."""
        component = manifest.get_component(self.component_name)
        if component is None:
            raise ContainerManifestInvalid("manifest has no `components.cntr` block")
        if component.schema_version not in self.supported_component_versions:
            raise ContainerManifestSchemaUnsupported(
                "Unsupported cntr component schema_version %r; this linktools-cntr supports %r"
                % (component.schema_version, self.supported_component_versions)
            )
        return component

    def load_and_get_component(
            self, repo_path: "PathType") -> "tuple[Optional[LinktoolsManifest], Optional[ManifestComponent]]":
        """``load()`` + ``get_component()`` in one step -- the structural
        gate every read-only consumer (Doctor, ``repo status``) needs before
        looking at a repository's manifest, so they can never drift on what
        counts as "this repo has no usable cntr manifest". ``(None, None)``
        means a project without a manifest file; any other problem is
        raised as ``ContainerManifestError``, same as ``load()``/
        ``get_component()`` individually."""
        manifest = self.load(repo_path)
        if manifest is None:
            return None, None
        return manifest, self.get_component(manifest)

    def ensure_loadable(self, manifest: "Optional[LinktoolsManifest]") -> None:
        """Raise if ``manifest`` (already statically valid) fails the cntr
        component gate or a host requirement -- called right before
        importing a repository's ``container.py``. ``None`` (a project
        without a manifest) is a no-op."""
        if manifest is None:
            return
        component = self.get_component(manifest)
        issues = self._check(manifest, component, phase="host")
        if issues:
            details = "; ".join(issue.message for issue in issues)
            raise ContainerIncompatible(f"Repository is incompatible with this host: {details}")

    def check_host_requirements(self, manifest: "Optional[LinktoolsManifest]") -> "list[RequirementResult]":
        """``python``/``package:linktools-cntr`` (plus any other project- or
        cntr-component-declared requirement) -- checked before a repository's
        Python is imported. An unrecognized requirement key fails closed:
        the manifest declared something this cntr version can't verify, so
        compatibility must never be assumed."""
        if manifest is None:
            return []
        component = manifest.get_component(self.component_name)
        if component is None:
            return []
        return self._check(manifest, component, phase="host")

    def check_runtime_requirements(self, manifest: "Optional[LinktoolsManifest]") -> "list[RequirementResult]":
        """``runtime:docker-engine``/``runtime:docker-compose`` -- checked
        before an actual Compose operation (doctor, repo status --runtime,
        plan, up/restart/compose preflight), never required just to add or
        load a repository."""
        if manifest is None:
            return []
        component = manifest.get_component(self.component_name)
        if component is None:
            return []
        return self._check(manifest, component, phase="runtime")

    def _check(self, manifest: "LinktoolsManifest", component: "ManifestComponent",
              phase: str) -> "list[RequirementResult]":
        results = self.registry.check(manifest.requires, phase=phase)
        results += self.registry.check(component.requires, phase=phase)
        return [r for r in results if r.status != RequirementStatus.SATISFIED]

    def _resolve_docker_engine(self, key: str) -> "Optional[str]":
        inspector = getattr(self.manager, "docker_inspector", None)
        if inspector is None:
            return None
        try:
            engine = inspector.get_engine_version()
            return (engine.server or engine.client) if engine else None
        except Exception:  # noqa: BLE001 - a version probe must never crash the caller
            return None

    def _resolve_docker_compose(self, key: str) -> "Optional[str]":
        inspector = getattr(self.manager, "docker_inspector", None)
        if inspector is None:
            return None
        try:
            return inspector.get_compose_version()
        except Exception:  # noqa: BLE001 - a version probe must never crash the caller
            return None
