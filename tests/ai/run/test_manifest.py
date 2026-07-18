#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for run.manifest: ExecutionManifest + sub-manifests, the revision /
fingerprint helpers, build_execution_manifest, compute_resumability, and the
JSON round-trip. Pure data/function checks -- no I/O."""

import pytest

from linktools.ai.run.manifest import (
    NON_RESUMABLE_EPHEMERAL_PROVIDER,
    NON_RESUMABLE_MISSING_RESOURCE_SNAPSHOT,
    NON_RESUMABLE_UNVERSIONED_HANDLER,
    DefaultManifestResolver,
    ExecutionManifest,
    MCPManifest,
    ManifestResolver,
    ResolvedExecution,
    ResourceRevision,
    Resumability,
    SCHEMA_VERSION,
    ToolManifest,
    VersionedProvider,
    build_execution_manifest,
    compute_resumability,
    descriptor_fingerprint,
    handler_revision,
    manifest_from_dict,
    manifest_to_dict,
    provider_revision,
)
from linktools.ai.errors import ManifestDriftError


class _ToolRef:
    def __init__(self, kind, name, config):
        self.kind = kind
        self.name = name
        self.config = config


class _Spec:
    def __init__(self, *, tools=(), primary="test-model", spec_id="agent-1"):
        self.id = spec_id
        self.tools = tools
        self.model = type("M", (), {"primary": primary})


# --- descriptor_fingerprint --------------------------------------------------


def test_descriptor_fingerprint_is_deterministic():
    a = descriptor_fingerprint(_ToolRef("builtin", "t", {"x": 1}))
    b = descriptor_fingerprint(_ToolRef("builtin", "t", {"x": 1}))
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_descriptor_fingerprint_differs_on_any_identity_field():
    base = descriptor_fingerprint(_ToolRef("builtin", "t", {"x": 1}))
    assert descriptor_fingerprint(_ToolRef("builtin", "t", {"x": 2})) != base
    assert descriptor_fingerprint(_ToolRef("builtin", "t2", {"x": 1})) != base
    assert descriptor_fingerprint(_ToolRef("mcp", "t", {"x": 1})) != base


# --- handler_revision --------------------------------------------------------


def test_handler_revision_uses_explicit_attribute_when_present():
    def handler(*a, **k):  # noqa: ARG001
        ...

    handler.revision = "v1.2.3"
    assert handler_revision(handler) == "v1.2.3"


def test_handler_revision_falls_back_to_module_qualname():
    def handler(*a, **k):  # noqa: ARG001
        ...

    revision = handler_revision(handler)
    assert revision is not None
    assert __name__ in revision
    assert "handler" in revision


def test_handler_revision_none_when_not_stably_versionable():
    # A bare object with no __module__/__qualname__ cannot be versioned.
    assert handler_revision(42) is None
    assert handler_revision("not a handler") is None


# --- provider_revision + VersionedProvider -----------------------------------


class _VersionedProvider:
    @property
    def revision(self) -> str:
        return "deploy-abc"


class _BareProvider:
    pass


def test_provider_revision_reads_revision_property():
    assert provider_revision(_VersionedProvider()) == "deploy-abc"


def test_provider_revision_none_for_unversioned_provider():
    assert provider_revision(_BareProvider()) is None
    assert provider_revision(None) is None


def test_versioned_provider_protocol_runtime_checkable():
    assert isinstance(_VersionedProvider(), VersionedProvider)
    assert not isinstance(_BareProvider(), VersionedProvider)


# --- build_execution_manifest ------------------------------------------------


def test_build_manifest_captures_spec_level_fields():
    spec = _Spec(tools=(_ToolRef("builtin", "t1", {}), _ToolRef("mcp", "t2", {"k": 1})))
    manifest = build_execution_manifest(
        spec, runnable_type="agent", runnable_fingerprint="fp"
    )
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.runnable_id == "agent-1"
    assert manifest.runnable_type == "agent"
    assert manifest.runnable_revision == "fp"
    assert manifest.runnable_fingerprint == "fp"
    assert manifest.model_name == "test-model"
    assert len(manifest.tool_descriptors) == 2
    names = [t.name for t in manifest.tool_descriptors]
    assert names == ["t1", "t2"]
    # Each tool carries a descriptor fingerprint; handler revisions are None
    # when no handlers were supplied (layered in from compiled run later).
    for tool in manifest.tool_descriptors:
        assert tool.descriptor_fingerprint
        assert tool.handler_revision is None


def test_build_manifest_records_handler_revisions_when_handlers_supplied():
    def handler_a(*a, **k):  # noqa: ARG001
        ...

    handler_a.revision = "ha-v1"

    spec = _Spec(tools=(_ToolRef("builtin", "t1", {}),))
    manifest = build_execution_manifest(
        spec,
        runnable_type="agent",
        runnable_fingerprint="fp",
        tool_handlers={"t1": handler_a},
    )
    assert manifest.tool_descriptors[0].handler_revision == "ha-v1"


def test_build_manifest_records_provider_revision_when_supplied():
    spec = _Spec()
    manifest = build_execution_manifest(
        spec,
        runnable_type="agent",
        runnable_fingerprint="fp",
        model_provider=_VersionedProvider(),
    )
    assert manifest.model_revision == "deploy-abc"
    assert manifest.model_provider == "_VersionedProvider"


# --- JSON round-trip (§13.8) -------------------------------------------------


def test_manifest_round_trips_through_dict():
    manifest = ExecutionManifest(
        schema_version=SCHEMA_VERSION,
        runnable_id="agent-1",
        runnable_type="agent",
        runnable_revision="fp",
        runnable_fingerprint="fp",
        model_name="m",
        model_provider="P",
        model_revision="pr",
        tool_descriptors=(
            ToolManifest(name="t", descriptor_fingerprint="d", handler_revision="h"),
        ),
        skill_revisions=(
            ResourceRevision(path="s.md", revision="r", etag="e", sha256="sh", artifact_id="a"),
        ),
        subagent_revisions=(
            ResourceRevision(path="sub.md", revision=None, etag=None, sha256=None, artifact_id="aid"),
        ),
        mcp_servers=(MCPManifest(name="srv", revision="mr"),),
        policy_revision="pol",
        security_baseline_revision="sec",
        capability_revision="cap",
        output_schema_id="osid",
        output_schema_revision="osrev",
    )
    restored = manifest_from_dict(manifest_to_dict(manifest))
    assert restored == manifest


def test_manifest_from_dict_tolerates_partial_legacy_payload():
    # An older/partial manifest mapping must still load.
    restored = manifest_from_dict(
        {"runnable_id": "a", "runnable_type": "agent", "schema_version": 1}
    )
    assert restored.runnable_id == "a"
    assert restored.tool_descriptors == ()
    assert restored.model_revision is None


# --- compute_resumability (§13.7 / §13.8) ------------------------------------


def _manifest(**overrides):
    base = dict(
        schema_version=SCHEMA_VERSION,
        runnable_id="a",
        runnable_type="agent",
        runnable_revision="fp",
        runnable_fingerprint="fp",
        model_name="m",
        model_provider="P",
        model_revision="pr",
    )
    base.update(overrides)
    return ExecutionManifest(**base)


def test_compute_resumability_resumable_when_versioned():
    manifest = _manifest(
        tool_descriptors=(
            ToolManifest(name="t", descriptor_fingerprint="d", handler_revision="h"),
        ),
    )
    verdict, reasons = compute_resumability(manifest)
    assert verdict is Resumability.RESUMABLE
    assert reasons == ()


def test_compute_resumability_non_resumable_for_unversioned_handler():
    manifest = _manifest(
        tool_descriptors=(
            ToolManifest(name="t", descriptor_fingerprint="d", handler_revision=None),
        ),
    )
    verdict, reasons = compute_resumability(manifest)
    assert verdict is Resumability.NON_RESUMABLE
    assert NON_RESUMABLE_UNVERSIONED_HANDLER in reasons


def test_compute_resumability_non_resumable_for_ephemeral_provider():
    # A named provider with no revision is ephemeral.
    manifest = _manifest(model_revision=None)
    verdict, reasons = compute_resumability(manifest)
    assert verdict is Resumability.NON_RESUMABLE
    assert NON_RESUMABLE_EPHEMERAL_PROVIDER in reasons


def test_compute_resumability_non_resumable_for_missing_resource_snapshot():
    manifest = _manifest(
        skill_revisions=(
            ResourceRevision(path="s.md", revision=None, etag=None, sha256=None, artifact_id=None),
        ),
    )
    verdict, reasons = compute_resumability(manifest)
    assert verdict is Resumability.NON_RESUMABLE
    assert NON_RESUMABLE_MISSING_RESOURCE_SNAPSHOT in reasons


def test_compute_resumability_resumable_when_resource_pinned_by_etag_only():
    # A single pinning field is enough: an etag-only snapshot (e.g. an
    # HTTP-served skill that exposes ETag but no content hash) is resumable.
    manifest = _manifest(
        skill_revisions=(
            ResourceRevision(path="s.md", revision=None, etag="etag-x", sha256=None, artifact_id=None),
        ),
    )
    verdict, reasons = compute_resumability(manifest)
    assert verdict is Resumability.RESUMABLE
    assert NON_RESUMABLE_MISSING_RESOURCE_SNAPSHOT not in reasons


def test_compute_resumability_reasons_are_deduplicated():
    manifest = _manifest(
        model_revision=None,
        tool_descriptors=(
            ToolManifest(name="t1", descriptor_fingerprint="d", handler_revision=None),
            ToolManifest(name="t2", descriptor_fingerprint="d2", handler_revision=None),
        ),
    )
    _, reasons = compute_resumability(manifest)
    # Two unversioned handlers collapse to one reason entry.
    assert reasons.count(NON_RESUMABLE_UNVERSIONED_HANDLER) == 1


# --- ManifestResolver Protocol -----------------------------------------------


def test_manifest_resolver_is_runtime_checkable():
    class _Resolver:
        async def resolve(self, manifest, *, spec):  # noqa: ARG002
            return ResolvedExecution(manifest=manifest)

    assert isinstance(_Resolver(), ManifestResolver)


def test_resolved_execution_default_notes_empty():
    manifest = _manifest()
    resolved = ResolvedExecution(manifest=manifest)
    assert resolved.notes == ()
    assert resolved.manifest is manifest


# --- DefaultManifestResolver (§13.6 drift detection) -------------------------


def test_default_resolver_accepts_matching_provider_revision():
    import asyncio

    manifest = _manifest(model_revision="rev-aaa")

    async def current(_name):
        return "rev-aaa"

    resolved = asyncio.run(DefaultManifestResolver(current).resolve(manifest, spec=None))
    assert "provider-revision-verified" in resolved.notes


def test_default_resolver_refuses_drifted_provider_revision():
    import asyncio

    manifest = _manifest(model_revision="rev-aaa")

    async def current(_name):
        return "rev-bbb"  # drifted

    with pytest.raises(ManifestDriftError):
        asyncio.run(DefaultManifestResolver(current).resolve(manifest, spec=None))


def test_default_resolver_refuses_unresolvable_provider():
    import asyncio

    manifest = _manifest(model_revision="rev-aaa")

    async def current(_name):
        return None  # model no longer resolvable

    with pytest.raises(ManifestDriftError):
        asyncio.run(DefaultManifestResolver(current).resolve(manifest, spec=None))


def test_default_resolver_skips_when_no_pinned_revision():
    import asyncio

    manifest = _manifest(model_revision=None)  # nothing pinned

    async def current(_name):
        return "rev-anything"

    resolved = asyncio.run(DefaultManifestResolver(current).resolve(manifest, spec=None))
    assert resolved.notes == ()  # no check performed, no note
