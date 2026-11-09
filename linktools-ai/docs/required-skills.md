# Required Skills and actual-read evidence

`RuntimeFeatures.agent(..., required_skills=("policy",), allow_skills=("policy", "optional"))`
requires the host to load `policy` before the first model request. `AgentSpec`
and Agent Markdown support the same `required_skills` field (`required-skills`
is an alias). Required IDs must be explicit, unique and included in `allow_skills`;
`"*"` is not a required ID. Existing `allow_skills` alone still discovers and
loads current Skills on demand. It is not evidence that a Skill was loaded.

## Original execution input

The admitted required root is represented by `SkillInput`: the named Skill
identity and its root `SkillSourceRef.declaration` `AssetVersionRef`. The Asset
reference includes source layer, immutable entry revision, size and digest.
Named Skill revision alone is not a content version.

`ExecutionInputContext.required_skills` preserves these descriptors in the
original execution input. The host verifies and reads the exact retained bytes,
then injects the normalized Skill instructions before calling the model. A
missing, filtered, unverifiable or unavailable required root fails closed. An
imported context must match the new execution's admitted required selection;
caller-supplied descriptors do not authorize arbitrary Asset reads. Recovery of
an already admitted execution retains its original descriptors and does not
replace them with the current directory contents.

Only required root instructions are frozen. A subsequent `load_skill("policy")`
returns that root and advertises **current** auxiliary resources with
`resource_status="current"`. A currently available package may have an empty
resource list. If no current package is available, the pinned root remains
usable and the resource view reports `resource_status="unavailable"`.
`load_skill("policy", path="...")` and ordinary, non-required Skills retain their
existing live behavior. Each successful resource read describes the version
actually read, not the required root's version.

## Asset retention and permissions

Required bytes stay with the Asset owner. `RetainedAssetReader` is an optional
read capability; existing custom readers need not implement it for ordinary
Skills. `AssetStore.require_retained_versions` checks each reference's actual
layer, reads the exact version and validates its size and digest. SQL Asset
history and Asset snapshots retain old versions. An in-memory backend retains
history only for that backend object's lifetime; the host must keep the owner
available for every recovery attempt. A pure `DirectoryAssetBackend` does not
retain history and is rejected for required roots even when the current file
still matches.

For directory inputs, publish an existing Asset snapshot to a host-owned
ObjectStore and configure `AssetStore.from_snapshot(...)`, or use an existing
SQL Asset owner. Keep that snapshot/history owner for the execution's lifetime.
Runtime does not silently copy these bytes to its own ObjectStore and does not
create a second freeze or garbage-collection lifecycle.

Admission uses the existing `EXECUTION_RUN` principal authorization and the
admitted Agent's `allow_skills`/host filter scope. The filter is not a per-asset
principal ACL. More detailed Asset authorization remains the host's
responsibility. Runtime configuration/discovery can parse declarations before
execution authorization; denied execution prevents the subsequent required
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

Required preloads are instructions, not synthetic tool calls. Their receipts are
attached to the actual ModelRequest metadata only after verifying the matching
instruction parts. With `include_content=True`,
`History.read_model_interactions(...)` exposes them in
`item.request["required_skill_reads"]` for the exact request. Metadata is not
added to model text. A recovery request can have its own real delivery receipt;
already recorded requests are not rewritten. Querying declarations or allowed
IDs alone never yields a successful-read receipt.
