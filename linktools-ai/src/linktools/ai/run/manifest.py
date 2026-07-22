#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ExecutionManifest: the immutable record of WHAT a run was prepared against --
the revisions / fingerprints of the runnable, model provider, tool handlers,
skills/subagents, MCP servers, policy, and capabilities. Persisted into the
RunDefinitionSnapshot at prepare time so resume can detect environment drift
instead of silently re-resolving "latest".

This module is the contracts + construction + provider-drift layer for
deterministic resume. The manifest is BUILT and PERSISTED at prepare time
(run/preparation.py runs after compile, so the resolved provider revision is
captured). The concrete ``DefaultManifestResolver`` consumes it on resume
(refuse drift, never fall back to latest) and is wired into
``Runtime.resume``. Tool-handler revisions and pinned asset (skill /
subagent) snapshots are NOT yet populated -- handlers are resolved at
execution time, not prepare time, so their revisions layer in once tool
resolution moves earlier; asset snapshotting is its own follow-up.

Revision helpers:
- ``descriptor_fingerprint`` -- content hash of a tool declaration (the ToolRef
  at prepare time, or a resolved descriptor post-compile).
- ``handler_revision`` -- a tool handler's revision: an explicit ``revision``
  attribute if the handler exposes one, else derived from its module / qualname
  / package version. Returns None for a handler that cannot be stably
  versioned.
- ``provider_revision`` -- a model provider's revision if it is a
  ``VersionedProvider``, else None (an unversioned provider makes the run
  non-resumable)."""

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping, Protocol, runtime_checkable

from ..errors import ManifestDriftError
from ..json import canonical_json


# --- resumability ------------------------------------------------------------


class Resumability(str, Enum):
    """Whether a run may be resumed. Set at creation time so an unversionable
    run is rejected up-front rather than discovered at resume."""

    RESUMABLE = "resumable"
    NON_RESUMABLE = "non_resumable"


#: Reasons a run may be marked NON_RESUMABLE. Stored as a plain string
#: alongside the verdict for audit; not an enum because multiple may apply and
#: the set is open-ended.
NON_RESUMABLE_DYNAMIC_OUTPUT = "dynamic_output_type"
NON_RESUMABLE_UNVERSIONED_HANDLER = "unversioned_handler"
NON_RESUMABLE_EPHEMERAL_PROVIDER = "ephemeral_provider"
NON_RESUMABLE_MISSING_RESOURCE_SNAPSHOT = "missing_resource_snapshot"


# --- VersionedProvider -------------------------------------------------------


@runtime_checkable
class VersionedProvider(Protocol):
    """A model provider that exposes a stable revision string (a config hash,
    deployment id, model version, or implementation version -- never just a
    bare vendor name like ``"openai"``). Providers that do not implement this
    cannot back a resumable run."""

    @property
    def revision(self) -> str: ...


def provider_revision(provider: Any) -> "str | None":
    """Return ``provider.revision`` if it is a non-empty string, else None.

    Accepts any object; a provider need only expose a ``revision`` attribute or
    property. A None result means the provider is unversioned and the run is
    not resumable."""
    revision = getattr(provider, "revision", None)
    if isinstance(revision, str) and revision:
        return revision
    return None


# --- revision / fingerprint helpers ------------------------------------------


def descriptor_fingerprint(descriptor: Any) -> str:
    """Stable content hash of a tool declaration. Operates on whatever the
    caller has at the time: a spec-level ToolRef (pre-compile) or a resolved
    descriptor (post-compile). The fingerprint covers the declaration's
    identity-defining fields (kind / name / config) so two declarations that
    differ in any of them fingerprint differently."""
    kind = getattr(descriptor, "kind", None)
    name = getattr(descriptor, "name", None)
    config = getattr(descriptor, "config", None)
    payload = canonical_json(
        {"kind": kind, "name": name, "config": _json_safe(config)}
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def handler_revision(handler: Any) -> "str | None":
    """A tool handler's revision. Preference order:

    1. an explicit ``revision`` attribute the handler exposes (downstream tools
       that know their own version);
    2. derived from ``module:qualname`` plus the enclosing package's version
       (``importlib.metadata``) when resolvable;
    3. None if neither yields a stable value -- such a handler cannot back a
       resumable run."""
    explicit = getattr(handler, "revision", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    module = getattr(handler, "__module__", None)
    qualname = getattr(handler, "__qualname__", None) or getattr(
        handler, "__name__", None
    )
    if not module or not qualname:
        return None
    pkg_version = _package_version(module)
    revision = f"{module}:{qualname}"
    if pkg_version is not None:
        revision = f"{revision}@{pkg_version}"
    return revision


def _package_version(module_name: str) -> "str | None":
    """Best-effort version of the top-level package owning ``module_name``."""
    try:
        import importlib.metadata as ilm  # noqa: D

        top = module_name.split(".")[0]
        return ilm.version(top)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    """Reduce arbitrary values to a canonical-JSON-safe shape so
    descriptor_fingerprint is deterministic regardless of the config's types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


# --- manifest dataclasses ----------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolManifest:
    """One tool's name + descriptor fingerprint + handler revision. The
    fingerprint identifies the declaration; the handler revision identifies the
    implementation that will execute it."""

    name: str
    descriptor_fingerprint: str
    handler_revision: "str | None"


@dataclass(frozen=True, slots=True)
class CapabilityRevision:
    """A pinned skill / subagent asset: its path + revision + etag +
    sha256 + artifact_id, so resume restores the EXACT asset instead of
    re-reading latest."""

    path: str
    revision: "str | None"
    etag: "str | None"
    sha256: "str | None"
    artifact_id: "str | None"


@dataclass(frozen=True, slots=True)
class MCPManifest:
    """One MCP server's name + revision, so an MCP-backed tool change is
    detectable on resume."""

    name: str
    revision: "str | None"


@dataclass(frozen=True, slots=True)
class ExecutionManifest:
    """The immutable record of what a run was prepared against. Persisted at
    prepare time; consumed by ``DefaultManifestResolver`` on resume to refuse
    drift. Provider revision is populated from the compiled run; tool-handler
    revisions are populated when the corresponding providers expose stable
    revisions; unversioned assets are marked non-resumable."""

    schema_version: int
    runnable_id: str
    runnable_type: str
    runnable_revision: "str | None"
    runnable_fingerprint: "str | None"
    model_name: "str | None"
    model_provider: "str | None"
    model_revision: "str | None"
    tool_descriptors: "tuple[ToolManifest, ...]" = field(default_factory=tuple)
    skill_revisions: "tuple[CapabilityRevision, ...]" = field(default_factory=tuple)
    subagent_revisions: "tuple[CapabilityRevision, ...]" = field(default_factory=tuple)
    mcp_servers: "tuple[MCPManifest, ...]" = field(default_factory=tuple)
    policy_revision: "str | None" = None
    security_baseline_revision: "str | None" = None
    capability_revision: "str | None" = None
    output_schema_id: "str | None" = None
    output_schema_revision: "str | None" = None


SCHEMA_VERSION = 1


# --- serialization (round-trip for persistence) ------------------------------


def manifest_to_dict(manifest: ExecutionManifest) -> "dict[str, Any]":
    """JSON-safe mapping for persistence in RunDefinitionSnapshot.manifest.
    Tuples become lists; round-trips via ``manifest_from_dict``."""
    return {
        "schema_version": manifest.schema_version,
        "runnable_id": manifest.runnable_id,
        "runnable_type": manifest.runnable_type,
        "runnable_revision": manifest.runnable_revision,
        "runnable_fingerprint": manifest.runnable_fingerprint,
        "model_name": manifest.model_name,
        "model_provider": manifest.model_provider,
        "model_revision": manifest.model_revision,
        "tool_descriptors": [
            {
                "name": t.name,
                "descriptor_fingerprint": t.descriptor_fingerprint,
                "handler_revision": t.handler_revision,
            }
            for t in manifest.tool_descriptors
        ],
        "skill_revisions": [_capability_revision_to_dict(r) for r in manifest.skill_revisions],
        "subagent_revisions": [
            _capability_revision_to_dict(r) for r in manifest.subagent_revisions
        ],
        "mcp_servers": [
            {"name": m.name, "revision": m.revision} for m in manifest.mcp_servers
        ],
        "policy_revision": manifest.policy_revision,
        "security_baseline_revision": manifest.security_baseline_revision,
        "capability_revision": manifest.capability_revision,
        "output_schema_id": manifest.output_schema_id,
        "output_schema_revision": manifest.output_schema_revision,
    }


def _capability_revision_to_dict(asset: CapabilityRevision) -> "dict[str, Any]":
    return {
        "path": asset.path,
        "revision": asset.revision,
        "etag": asset.etag,
        "sha256": asset.sha256,
        "artifact_id": asset.artifact_id,
    }


def manifest_from_dict(data: "Mapping[str, Any]") -> ExecutionManifest:
    """Reconstruct an ExecutionManifest from its persisted mapping. Tolerant of
    older partial manifests (missing keys default to None / empty)."""
    return ExecutionManifest(
        schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        runnable_id=data.get("runnable_id") or "",
        runnable_type=data.get("runnable_type") or "",
        runnable_revision=data.get("runnable_revision"),
        runnable_fingerprint=data.get("runnable_fingerprint"),
        model_name=data.get("model_name"),
        model_provider=data.get("model_provider"),
        model_revision=data.get("model_revision"),
        tool_descriptors=tuple(
            ToolManifest(
                name=t["name"],
                descriptor_fingerprint=t["descriptor_fingerprint"],
                handler_revision=t.get("handler_revision"),
            )
            for t in data.get("tool_descriptors", ())
        ),
        skill_revisions=tuple(
            _capability_revision_from_dict(r) for r in data.get("skill_revisions", ())
        ),
        subagent_revisions=tuple(
            _capability_revision_from_dict(r) for r in data.get("subagent_revisions", ())
        ),
        mcp_servers=tuple(
            MCPManifest(name=m["name"], revision=m.get("revision"))
            for m in data.get("mcp_servers", ())
        ),
        policy_revision=data.get("policy_revision"),
        security_baseline_revision=data.get("security_baseline_revision"),
        capability_revision=data.get("capability_revision"),
        output_schema_id=data.get("output_schema_id"),
        output_schema_revision=data.get("output_schema_revision"),
    )


def _capability_revision_from_dict(data: "Mapping[str, Any]") -> CapabilityRevision:
    return CapabilityRevision(
        path=data.get("path") or "",
        revision=data.get("revision"),
        etag=data.get("etag"),
        sha256=data.get("sha256"),
        artifact_id=data.get("artifact_id"),
    )


# --- construction -----------------------------------------------------------


def build_execution_manifest(
    spec: Any,
    *,
    runnable_type: str,
    runnable_fingerprint: "str | None",
    model_provider: Any = None,
    tool_handlers: "Mapping[str, Any] | None" = None,
    policy_revision: "str | None" = None,
    security_baseline_revision: "str | None" = None,
    capability_revision: "str | None" = None,
    skill_revisions: "tuple[CapabilityRevision, ...]" = (),
    subagent_revisions: "tuple[CapabilityRevision, ...]" = (),
    mcp_servers: "tuple[MCPManifest, ...]" = (),
    output_schema_id: "str | None" = None,
    output_schema_revision: "str | None" = None,
) -> ExecutionManifest:
    """Construct an ExecutionManifest from the available inputs.

    ``spec`` supplies the runnable id, the model name (``spec.model.primary``),
    and the tool declarations (``spec.tools`` -- spec-level ToolRefs).
    ``model_provider`` is the resolved model bundle (its revision is recorded
    when supplied; the prepare path passes the compiled bundle). ``tool_handlers``
    maps tool name -> handler; when supplied, each tool's handler_revision is
    computed, otherwise handler_revision is None. Handlers are resolved at
    execution time today, so the prepare path supplies none and tool-handler
    revisions stay None until tool resolution moves earlier.
    """
    model_policy = getattr(spec, "model", None)
    model_name = getattr(model_policy, "primary", None)
    tool_refs = getattr(spec, "tools", None) or ()
    handlers = dict(tool_handlers) if tool_handlers is not None else {}
    tool_descriptors = tuple(
        ToolManifest(
            name=ref.name,
            descriptor_fingerprint=descriptor_fingerprint(ref),
            handler_revision=(
                handler_revision(handlers[ref.name])
                if ref.name in handlers
                else None
            ),
        )
        for ref in tool_refs
    )
    return ExecutionManifest(
        schema_version=SCHEMA_VERSION,
        runnable_id=getattr(spec, "id", "") or "",
        runnable_type=runnable_type,
        # runnable_revision aliases the content fingerprint until a separate
        # semantic-revision source (e.g. an explicit spec revision / semver)
        # exists; the two fields stay distinct so a future source can fill the
        # revision without changing the fingerprint.
        runnable_revision=runnable_fingerprint,
        runnable_fingerprint=runnable_fingerprint,
        model_name=model_name,
        model_provider=type(model_provider).__name__ if model_provider is not None else None,
        model_revision=provider_revision(model_provider),
        tool_descriptors=tool_descriptors,
        skill_revisions=skill_revisions,
        subagent_revisions=subagent_revisions,
        mcp_servers=mcp_servers,
        policy_revision=policy_revision,
        security_baseline_revision=security_baseline_revision,
        capability_revision=capability_revision,
        output_schema_id=output_schema_id,
        output_schema_revision=output_schema_revision,
    )


def compute_resumability(
    manifest: ExecutionManifest,
) -> "tuple[Resumability, tuple[str, ...]]":
    """Pure verdict over a manifest: RESUMABLE unless any disqualifier is
    present -- an unversioned tool handler, an ephemeral/unversioned provider,
    or a asset snapshot missing all pinning fields. Returns the verdict plus
    the de-duplicated reasons that applied (empty for RESUMABLE).

    A None ``handler_revision`` / ``model_revision`` only disqualifies when the
    corresponding component is PRESENT in the manifest (a tool whose handler
    revision was not recorded; a named provider with no revision). This keeps
    the verdict honest about what the manifest actually claims.

    A asset (skill / subagent) is considered pinned when ANY of its
    snapshot fields is set (sha256 / revision / artifact_id / etag) -- a single
    pinning field is enough. MCP-server drift is detected by the
    resume-side resolver, not by this verdict.

    Dynamic-output-type detection requires inspecting the compiled
    output type; it is not inferable from this manifest alone and is layered in
    once the manifest carries compiled revisions."""
    reasons: list = []
    for tool in manifest.tool_descriptors:
        if tool.handler_revision is None:
            reasons.append(NON_RESUMABLE_UNVERSIONED_HANDLER)
    if manifest.model_name is not None and manifest.model_revision is None:
        reasons.append(NON_RESUMABLE_EPHEMERAL_PROVIDER)
    for asset in (*manifest.skill_revisions, *manifest.subagent_revisions):
        if (
            not asset.sha256
            and not asset.revision
            and not asset.artifact_id
            and not asset.etag
        ):
            reasons.append(NON_RESUMABLE_MISSING_RESOURCE_SNAPSHOT)
    if reasons:
        return Resumability.NON_RESUMABLE, tuple(dict.fromkeys(reasons))
    return Resumability.RESUMABLE, ()


# --- ManifestResolver --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedExecution:
    """The resolved, drift-checked execution for a manifest. Carried out of
    ``ManifestResolver.resolve`` to feed the resume path."""

    manifest: ExecutionManifest
    notes: "tuple[str, ...]" = field(default_factory=tuple)


@runtime_checkable
class ManifestResolver(Protocol):
    """Resolve a persisted manifest back to an executable form on resume.

    Contract: the exact pinned versions must be present and fingerprint-consistent;
    a missing version or fingerprint mismatch is an explicit failure. Silent
    fallback to "latest" is forbidden. ``spec`` is the run's deserialized spec
    (the manifest pins identity; the spec is what gets re-resolved against the
    current environment)."""

    async def resolve(
        self,
        manifest: ExecutionManifest,
        *,
        spec: Any,
    ) -> ResolvedExecution: ...


class DefaultManifestResolver:
    """Provider-revision drift detection.

    Re-resolves the manifest's declared model against the current environment
    and refuses (``ManifestDriftError``) when the provider revision changed
    between prepare and resume, or when the model is no longer resolvable.
    Drift checks for tool handlers and pinned assets are layered in once
    those revisions are populated at prepare time.

    ``resolve_model_revision`` maps a model name to its CURRENT revision (or
    None if unresolvable), so this class stays decoupled from the model layer.
    A manifest with no recorded provider revision (``model_revision`` is None)
    is skipped -- there is nothing to check, so resume proceeds (no silent
    fallback to "latest", just no pin to compare against)."""

    def __init__(
        self,
        resolve_model_revision: "Callable[[str], Awaitable[str | None]]",
    ) -> None:
        self._resolve_model_revision = resolve_model_revision

    async def resolve(
        self,
        manifest: ExecutionManifest,
        *,
        spec: Any,
    ) -> ResolvedExecution:
        del spec  # unused: drift checks below key off the manifest, not the spec
        notes: list = []
        if manifest.model_revision is not None and manifest.model_name:
            current = await self._resolve_model_revision(manifest.model_name)
            if current is None:
                raise ManifestDriftError(
                    f"model {manifest.model_name!r} is no longer resolvable; "
                    f"refusing resume (manifest pinned revision "
                    f"{manifest.model_revision[:12]})"
                )
            if current != manifest.model_revision:
                raise ManifestDriftError(
                    f"model {manifest.model_name!r} revision drifted: manifest "
                    f"pinned {manifest.model_revision[:12]}, current "
                    f"{current[:12]}"
                )
            notes.append("provider-revision-verified")
        return ResolvedExecution(manifest=manifest, notes=tuple(notes))
