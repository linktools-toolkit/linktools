# Original execution input

Runtime supplies two read-only views on `AgentContext`, including contexts
passed to ordinary tools, registered capability hooks, and referenced-input
loaders:

- `original_input` is the existing accepted user-input view. It contains the
  original `prompt`, `files`, and `attachments`, in their accepted order. File
  descriptors retain their identifiers, MIME types, sizes, and digests. Asset
  references also retain the selected loader identity and reconstruction
  parameters. Ordinary file bodies are not added to this view.
- `input_context` is the existing `ExecutionInputContext`. Its original
  history, session metadata, memory capture, repository instructions, and
  preloaded Skill descriptors remain associated with that execution. Calling
  `model_messages()` returns detached messages rather than mutable stored
  history.

These are projections of the execution's existing durable owners, not another
input registry. Nested JSON values are detached on read. Mutating a returned
list or dictionary cannot change the accepted input. A manually constructed
`AgentContext` can omit either view, in which case that field is `None`; absence
must not be interpreted as an empty admitted input set.

`session_metadata` keeps its existing semantics. A native session's recovered
attempt can see current session metadata there while
`input_context.session_metadata` still contains the metadata accepted by the
original attempt. Imported contexts keep their imported metadata. Do not use
live session metadata to reconstruct original membership.

The views survive same-version `Runtime.handoff()` and reopen through the
existing execution context and stored-input codecs. Normal Runtime close
cancels unfinished execution; it is not a handoff operation. Recovery control
uses the recovery actor's authorization, while tools still receive the
originally admitted principal.

An input reference describes bytes and where they came from. Neither that
reference nor `source_parameters` proves current authorization. A controlled
loader must still apply its authorization policy. Historical tool output is
not initial user input merely because it appears in captured history. Its
original producer execution, run, request, and call must come from retained
producer facts; missing facts must not be filled with the current context.

Semantic output validators can use `ctx.deps.original_input` or
`ctx.deps.input_context` with the existing registered capability
`after_output_validate` hook. Raise the existing `ModelRetry` when correction
is needed and use the Agent's `output_retries` budget. Keep large membership
sets outside the output schema. The input view does not add a second validator
registry or change tool effect and retry ownership.

## Retained Rule instructions

Asset-backed Rule documents retain `asset_source_id` and the exact
`AssetVersionRef` alongside their accepted `source`, `scope`, and parsed body.
`RuleInstructionAdapter.decode_markdown()` is the shared public parser for
loading declarations and reconstructing their Markdown frontmatter.

Accepted execution-context, input-capture, and dynamic instruction-overlay
storage omit the Rule body only
when the configured source implements `RetainedAssetReader`, the exact version
is available, and parsing it reproduces the accepted source, scope, and body.
The stored document contains the Asset reference and the parsed body's SHA-256.
Reading verifies the Asset's original size/digest, parses that version, checks
source/scope/body digest, and restores the same public instruction payload.
The complete execution-context semantic digest is checked after reconstruction.
No current Asset head is consulted to replace a missing historical version.

Dynamic instruction overlays retain their existing checkpoint owner and use a
small database envelope containing the projected documents and the original
full instruction payload's semantic digest. The envelope's stored-byte digest
is checked separately. Recovery resolves the fixed Asset references before
checking that semantic digest. Tool-argument barrier digests and the approval
frontier's ordered source/scope/body digest are unchanged. Barrier compare-and-
swap, historical replay, and uncertain-write readback continue to select the
durable cumulative frontier rather than a newly resolved instruction set.

Runtime supplies its configured Asset readers automatically. Standalone
`TaskEngine.open(..., asset_sources={"rules": asset_store})`,
`InputCaptures.from_persistence(..., asset_sources={"rules": asset_store})`, and
`ExecutionContextArchive(..., asset_sources={"rules": asset_store})` accept the
same explicit source mapping. The key is the original `RuntimeFeatures` group
ID. A missing source or version fails closed when a stored reference needs it.
A reference is provenance, not a grant of additional Asset access; hosts remain
responsible for the borrowed reader's authorization and lifetime.

SQL Asset history and Asset snapshots can retain old versions independently of
current heads. In-memory Asset history lasts only for that backend's lifetime.
Directory-backed Rule files do not promise retention, even while their current
bytes match. These documents, Workspace `AGENTS.md`, and custom resolver
contents without verified retained provenance keep their accepted bodies in
Runtime records. Reopening them does not reread mutable source files. Ordinary
instruction text remains database content, not a Runtime ObjectStore file.

Persistence snapshots and garbage collection retain initial instructions
through the existing execution-context archive reference and dynamic overlays
through their checkpoint recovery record. Neither adds Runtime ObjectStore
files. Asset version bytes remain with their original Asset owner, as with
preloaded Skills and MCP resources. A Runtime snapshot does not include those external Asset bytes.
Retain or separately snapshot that Asset owner and supply it when reopening the
Runtime, reading a captured input, or replaying the context. Runtime does not
create a duplicate Asset version store or a new Asset cleanup lifecycle.

### Instruction and definition storage ownership

| Stored fact | Durable owner and source | Representation and recovery |
| --- | --- | --- |
| Accepted initial Rule instructions | Execution-context archive; resolved Rule Asset version | Verified retained sources use references. Source/scope/body integrity and the complete context are restored from the original Asset owner. |
| Captured Rule instructions and captured input context | Input-capture record; accepted execution context | The same retained-source projection is stored. Capture reads return the full accepted public payload through explicitly supplied Asset readers. |
| Workspace instructions and custom resolver documents | Accepted context/capture records; mutable Workspace or arbitrary resolver | Accepted body remains in the database. There is no independently retained Asset version to recover it from. |
| Pre-context admission instructions | Execution admission record; accepted resolver result | Body remains until an accepted context takes ownership. Records with `input_context` do not persist a second `repository_instructions` payload. |
| Dynamically discovered instruction overlay | Checkpoint recovery record; accepted tool-boundary frontier | Verified retained Rules use fixed Asset references; Workspace/custom bodies remain inline. An envelope preserves the original full semantic digest separately from stored-byte integrity. Recovery restores the original frontier before validating barriers and approvals. |
| Preloaded Skill root | Asset owner; `SkillInput` declaration version | Already reference-only. Recovery requires the retained Asset source. |
| Agent and Task bindings | Their execution/graph binding stores; compiled declarations | Accepted normalized contracts remain. They do not carry a sufficient Asset version plus compiler inputs to reconstruct the contract from raw authoring Assets. |
| Model contract and actual model requests | Execution/evaluation and model-history owners | Retained evaluation semantics and actual model-visible messages remain. A declaration reference alone does not recover what a provider actually received. |
