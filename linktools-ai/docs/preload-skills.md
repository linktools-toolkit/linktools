# Preloaded Skills and actual-read evidence

`RuntimeFeatures.agent(..., preload_skills=("policy",), allow_skills=("optional",))`
requires the host to load `policy` successfully before the first model request.
Preloading is mandatory: a missing or failed preload prevents the model call.
`AgentSpec` and Agent Markdown support the same `preload_skills` field (`preload-skills`
is an alias). Preloaded IDs must be explicit and unique; `"*"` is not a preloaded ID.
A preloaded Skill does not also need to appear in `allow_skills`: its root and
auxiliary resources are in scope automatically. The effective Skill scope is the
union of `allow_skills` and the explicit `preload_skills` IDs, still subject to the
host filter and execution authorization. Unlisted Skills are not added to scope.
Existing `allow_skills` alone still discovers and loads current Skills on demand. It is not evidence that a Skill was loaded.

## Original execution input

The admitted preloaded root is represented by `SkillInput`: the named Skill
identity and its root `SkillSourceRef.declaration` `AssetVersionRef`. The Asset
reference includes source layer, immutable entry revision, size and digest.
Named Skill revision alone is not a content version.

`ExecutionInputContext.preload_skills` preserves these descriptors in the
original execution input. The host verifies and reads the exact retained bytes,
then directly injects the normalized root body as an `InstructionPart` before
calling the model. This is not just registration or a receipt: every actual
model request verifies that the pinned body is present in its instruction parts.
Preloading does not automatically read the auxiliary resource directory. A
missing, filtered, unverifiable or unavailable preloaded root fails closed. An
imported context must match the new execution's admitted preloaded selection;
caller-supplied descriptors do not authorize arbitrary Asset reads. Recovery of
an already admitted execution retains its original descriptors and does not
replace them with the current directory contents.

Only preloaded root instructions are frozen. A subsequent `load_skill("policy")`
returns that root and advertises **current** auxiliary resources with
`resource_status="current"`. A currently available package may have an empty
resource list. If no current package is available, the pinned root remains
usable and the resource view reports `resource_status="unavailable"`.
`load_skill("policy", path="...")` and ordinary, non-preloaded Skills retain their
existing live behavior. Each successful resource read describes the version
actually read, not the preloaded root's version.

## Asset retention and permissions

Preloaded bytes stay with the Asset owner. `RetainedAssetReader` is an optional
read capability; existing custom readers need not implement it for ordinary
Skills. `AssetStore.require_retained_versions` checks each reference's actual
layer, reads the exact version and validates its size and digest. SQL Asset
history and Asset snapshots retain old versions. An in-memory backend retains
history only for that backend object's lifetime; the host must keep the owner
available for every recovery attempt. A pure `DirectoryAssetBackend` does not
retain history and is rejected for preloaded roots even when the current file
still matches.

For directory inputs, publish an existing Asset snapshot to a host-owned
ObjectStore and configure `AssetStore.from_snapshot(...)`, or use an existing
SQL Asset owner. Keep that snapshot/history owner for the execution's lifetime.
Runtime does not silently copy these bytes to its own ObjectStore and does not
create a second freeze or garbage-collection lifecycle.

Admission uses the existing `EXECUTION_RUN` principal authorization and the
admitted Agent's combined `allow_skills` and `preload_skills` scope, after the
host filter. The filter is not a per-asset
principal ACL. More detailed Asset authorization remains the host's
responsibility. Runtime configuration/discovery can parse declarations before
execution authorization; denied execution prevents the subsequent preloaded
retained-version capture and model request, not that earlier configuration read.

## Querying actual evidence

Successful `load_skill` results carry a `SkillReadReceipt` in native tool-return
metadata. `History.history(...)` projects it as `ExecutionHistoryItem.skill_read`,
with the existing execution, agent-run, model-request and tool-call filters.
The receipt includes source, resource path, exact Asset version when available,
and SHA-256 of the delivered content. For parsed root instructions, the delivery
digest may differ from the raw Asset digest. `identity_verified` is false for
successful inline/custom content without an observed Asset identity. Failed
reads retain their normal failed result and have no successful receipt.

Skill preloads are instructions, not synthetic tool calls. Their receipts are
attached to the actual ModelRequest metadata only after verifying the matching
instruction parts. With `include_content=True`,
`History.read_model_interactions(...)` exposes them in
`item.request["preload_skill_reads"]` for the exact request. Metadata is not
added to model text. A recovery request can have its own real delivery receipt;
already recorded requests are not rewritten. Querying declarations or allowed
IDs alone never yields a successful-read receipt.
