# File tools and model input

The model-facing candidates are `list_files`, `search_files`, `read_file`, and
`write_file`. Select only what the task needs using existing `allow_tools`,
`allow_skills`, and compiler `capability_filter`. Selection is not authorization.
No runtime enable-tools registry or implicit format converter is introduced.

| Task | Selected tools | Boundary |
| --- | --- | --- |
| Screenshot audit, complete manifest | read_file, search_files | Read-only; no commands or Skills |
| Screenshot audit requiring discovery | list_files, read_file, search_files | Only approved directories |
| Text editing | list_files, search_files, read_file, write_file | Existing write policy |
| Authorized format conversion with known paths | read_file, run_command, list_skills, load_skill | Select trusted Skills and existing CLI permissions |

The professional configuration selects `run_command` and `read_file` as tools,
plus a nonempty `allow_skills` to materialize `list_skills` and `load_skill`.
Add discovery/search only when required. `create_directory`, `file_info` and
the explicit host entry points remain available when deliberately selected.
An edit-only configuration keeps `edit_file`; it is not upgraded to whole-file
write permission. Broad `*` selectors and the implicit default Agent
remain broad (the full 14-tool catalog), not a minimal profile. The minimal writer
example explicitly selects the four modern file tools.

The OpenAI Chat, OpenAI Responses and Anthropic Messages adapters own their
protocol-specific input encoding. Declaring a MIME type does not make an adapter
support an otherwise unsupported wire format. File and business tools do not
encode provider-specific requests.

## What read_file can read

Recognition, transmission, and model understanding are separate decisions.
`inspect_file` uses a bounded authorized buffer with optional python-magic and
the system libmagic/database. Content detection is enabled only when both the
binding and native library/database initialize successfully. Otherwise it stays
disabled without repeated initialization attempts; ordinary reads/writes/lists
continue with honest extension/unknown hints. Install then restart to enable it.
An extension hint is not verified content.
Directory listings only give extension hints and never probe every body.
MIME recognition is not validation, safety approval, or proof of model support.

Bounded text previews do not require attachment permission. Binary files and
`as_file=True` require the configured model MIME permission and a supported
provider encoding. A refusal reports the relative path, size, MIME type and
reason without returning file contents. A batch remains atomic: one refused
item fails the call, rather than publishing a partial set of attachments.
PDF and Office files are not automatically converted or parsed as text.

* Text/log/CSV/JSON/XML and other recognized text use bounded UTF-8 previews.
  Invalid UTF-8 is replaced with `lossy=true`; an empty file remains readable.
* Images, PDF, audio, video, and other binary documents are attached only when the
  captured model route and actual SDK endpoint can encode them. Model declarations
  cannot make an unsupported SDK format encodable.
* `as_file=true` requests a whole input, including TXT/CSV. OpenAI Chat text-like
  binary inputs become `text-inline`; they are not native file blocks. A PDF
  file block is meaningful only for a route whose model supports PDF.
* Office, archive/container subtypes, and unfamiliar formats are not implicitly
  rendered, extracted, renamed, or unpacked. Use a selected trusted Skill and
  deployed CLI, or the host's existing authorized preparation stage.

All new model bindings use one `input_types` string list. Its empty default
rejects attachments; ordinary prompt text is unaffected. The names `image`,
`audio`, `document`, and `video` expand to the corresponding SDK Literal MIME
sets. Use `mime:` for exact MIME types or `type/*` families. There is no `text`
category; a text file can use `mime:text/plain`. Unmatched attachments are
rejected. A match does not bypass SDK encoding, workspace-provider authorization, resource
limits, or permissions.

The `vision` API argument and CLI flag are removed. Legacy environment or
workspace settings produce migration errors and must be replaced with
`input_types`. Recovering a historical execution requires a binding whose
contract matches its persisted historical contract; an ordinary new binding
cannot substitute its empty allowlist for that contract.

### Minimal model declaration

```python
from linktools.ai.model import create_model

binding = create_model(api="openai",
    model="your-model",
    input_types=["image", "document", "mime:audio/*", "mime:text/csv"],
)
```

Python and YAML use the same token list. Category expansion is resolved at
capture into the canonical `input_mime_types` list; restore uses those stored
MIME values without repeating category expansion against the current SDK.
A category includes the SDK's known types, whereas `mime:audio/*` permits the
whole audio family. Custom MIME values remain available through `mime:`.

The host is responsible for verifying that its provider/model accepts the
permitted MIME types. Transport selection remains internal; there is no separate
status, transport, unknown-input, or per-MIME byte-limit configuration. The
adapter still checks whether the actual SDK input can be encoded.

## Read contract

`read_file(path, as_file=False, offset=None, limit=None, cursor=None,
full_metadata=False)` accepts one path or a nonempty ordered array. Repeated
paths are distinct occurrences. `files` contains only attachments;
`texts` contains only preview text, each linked by `input_index`.
A failing member publishes neither earlier text nor attachments.

Range/cursor/metadata arguments require exactly one text preview (a singleton
array qualifies). Default range is 200 lines. `offset` is zero-based;
`cursor` is opaque and mutually exclusive with offset. Only `full_metadata=true`
scans for a complete raw-byte SHA-256 and line count. A bounded preview without
that hash still permits an ordinary exact edit without a version assertion.
`truncated` means a requested range hit a budget; `eof=false` alone only means
more text exists. A giant line can continue within the same line via cursor.
A changed source rejects an old cursor. Mutable stat identity is not an immutable
multi-file snapshot or an external-process atomic CAS.

All batch entries count against the existing input-part limit. Source IO, bytes,
output and elapsed time are bounded; shrink the batch/range when rejected. Media
is checked before whole-body allocation. Parallel tool attachments are checked
again against the final request's aggregate binary budget.

Attachment occurrences are accepted separately from included-in-request facts.
After one complete logical model interaction, binary bodies leave active context;
interruption and transport retry do not prematurely consume them. Raw history is
unchanged. Mixed results retain attachment identity; only same-hash/same-range
text previews can be deduplicated in the compressed view. Basic previews without
a full hash remain intact.

Direct binary inputs carry one generated JSON metadata text block followed by the
binary content. For the separate `files` argument, these pairs precede the prompt
in the original file-list order. Explicit content arrays keep their authored
text/file order. Workspace inputs use this same block instead of a separate path
label. Its fields are `attachment`, `size`, `mime_type`, and `path` when known.
`attachment` is the existing content identifier: file-tool binding uses the
Runtime occurrence ID, while other inputs retain their caller/SDK identifier.
URL or absolute-host-path identifiers are not echoed. Paths come from authorized
structured input metadata, never from parsing arbitrary user text.

Workspace `inspect_file` facts and current `read_file`/`attach_files` result
entries use `mime_type`; extension hints use `extension_mime_type`.
`read_file` retains `input_index` and `sha256` where applicable. `read_file` and
`attach_files` use behavior revision 1: when files are returned, the native tool
result is `[metadata, *files]`. Text-only results retain their metadata object.
Old stored tool results remain readable without rewriting their fields. SDK content retains its
`media_type` field at the adapter boundary.
Runtime binding adds the existing attachment identifier to each new `read_file`
file entry. These tool results already carry the metadata; the files stay in the
same native tool result, associated with its `tool_call_id`, rather than becoming
extra user content. Each provider uses its SDK's native tool-result mapping (and
its standard fallback when that protocol cannot carry files inside a tool result).

After a complete response, active context retains the existing metadata and
removes only the binary content blocks, omitting the resulting empty user content
block while preserving tool call/return pairing. No consumed-content prefix is
added. Old stored labels and markers remain readable without rewriting raw history. Legacy
binaries lacking a reliable metadata association receive one metadata text block
when projected. URL inputs and uploaded-file references retain their existing
lifetime. Recoverable reference inputs use the explicit contract below; ordinary
`BinaryContent` still retains its original bytes in durable history.

## Recoverable reference inputs

Use `ReferencedContent` when the host can reliably retrieve the **same immutable
content** again. This is a property of the source contract, not of an object-store
brand, file extension, size or currently valid URL. A mutable source that returns
its latest value is insufficient. If the accepted version is no longer available,
the read fails; it must not silently substitute current content.

The input carries an existing `AssetVersionRef`, its MIME type and the existing
`CapabilityRef` of the authorized loader. Runtime persists that descriptor, not
another copy of the Asset bytes. The capability reference identifies the loader's
behavior namespace by `id + revision`; the Asset reference identifies one resource
version within that namespace. Its `layer_id` is a storage layer, not another
capability identity. No URL, credentials, host path or callback digest is needed.

Register one loader on the selected Agent. It may route multiple configured
origins, but must not probe unrelated readers to find one that accepts a reference.
The following example uses `HostApp.read_admitted_version`, an application-owned
operation, not a LinkTools API:

```python
from linktools.ai.capability import AgentContext, ReferencedInputsCapability
from linktools.ai.model import ReferencedContent
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.spec import CapabilityRef

source = CapabilityRef("capability", "recoverable-inputs", 1)

async def load_input(ctx: AgentContext["HostApp"], item: ReferencedContent) -> bytes:
    return await ctx.app.read_admitted_version(
        principal=ctx.principal,
        execution_id=ctx.execution_id,
        ref=item.ref,
        media_type=item.media_type,
    )

features = RuntimeFeatures("documents")
features.capability(
    ReferencedInputsCapability(load_input),
    id=source.id, revision=source.revision, scope="agent",
)

# `ref` and `media_type` come from the same host-authorized immutable manifest.
prompt = ["Inspect this document", ReferencedContent(
    ref, media_type=media_type, source=source,
)]
```

`read_admitted_version` must authorize the original admitted execution principal
and that execution's selected input membership **before** source IO. Possessing
a reference, a signed URL, a reader or a service credential is not authorization.
The MIME type must agree with the same trusted manifest: `AssetVersionRef` does
not contain a MIME type. Restored or imported references still require this check,
and their source must match the selected capability revision before the loader is
called. The host owns same-revision configuration consistency. A model-authored
reference is not an admitted input.

The host performs a bounded exact-version read and owns cleanup of any temporary
download. Prefer memory when suitable; otherwise use a per-execution temporary
directory excluded from workspace archives and backups, and remove the materialized
file on success, failure and cancellation before returning bytes. This does not
promise cleanup after an uncatchable process termination. Runtime verifies the
returned bytes against the reference's existing size and SHA-256 integrity
contract. It does not create a second Asset store or generic URL resolver.

Hydration happens only in an outbound provider-message copy. Frozen input,
input capture, tool-result persistence, checkpoints and prepared request traces
keep the reference. Existing input-part and aggregate byte budgets count its known
size; the configured MIME and actual protocol must support it. Native OpenAI Chat,
OpenAI Responses and Anthropic Messages adapters encode the temporary
`BinaryContent`; there is no model-name protocol inference or automatic size-based
Files API upload. No additional retry or external-tool replay is introduced.
Token counting, compaction and recovery may call the loader again. It must be a
read-only, repeatable exact-version read, not a one-shot external side effect.

A business tool returns `ToolReturn(return_value=[metadata, reference])` to keep
an attachment in its native tool result. A direct reference return is also
supported. Reference items must be direct results or top-level list items, matching
the SDK's native multimodal result shape; references hidden in JSON mappings or
nested sequences are rejected rather than silently serialized as text.
`ToolReturn(content=[reference])` remains an explicit request for additional user
content; Runtime does not move it into a neighboring tool result. Arbitrary return
objects retain their ordinary JSON-snapshot contract. A descriptor-like snapshot
is plain result data, never promoted into a readable or recoverable attachment.
Persisted reference-bearing returns cannot introduce executable tools. Native
results participate in the same MIME checks, byte budgets and safe trace policy
as user inputs, including parallel tool results.
After a complete logical model response, the active context retains ordinary
summary text and the original descriptor in its durable metadata, and no longer
hydrates that occurrence. SDK serialization and provider messages expose only
the summary, not descriptor fields or loader reconstruction parameters. This is not an automatic
semantic content summary. Pending or interrupted references remain recoverable.

Even with `TraceCapability(record_http_bodies=True)`, a request containing a
reference omits its entire raw HTTP request and response bodies. Each attempt's
diagnostics records `body_omission_reason="referenced_content_lifecycle"`; status,
request IDs, timing, counts and stream observations remain available. This is a
declared retention omission, not a capture error, empty body or proof of stream
completion. Provider failures likewise expose `content_omission_reason` while
retaining exception types, relationships and frame locations without arbitrary
messages, notes, source lines or echoed file content. Subsequent ordinary requests
retain their configured capture behavior.

This is opt-in. Existing `BinaryContent`, workspace file inputs and already-uploaded
`UploadedFile` references keep their contracts. `UploadedFile` names an existing
provider file; Runtime does not upload or delete it. Unrecoverable originals and
intentional long-lived outputs still require their existing storage policy. Old
history is not rewritten or purged, and moving an input into an output directory
does not make it an independently authorized operational artifact.

## Discovery and edits

`list_files(path=".", include_glob="*")` lists direct children; `**` explicitly
allows recursion. `search_files(pattern, path=".", include_glob=None)` searches
text recursively, but uses the same glob rule for each candidate relative to path:
`*` does not cross `/`; `**` spans zero or more levels; `**/*.py` includes root
files. Existing hidden/ignored authorization semantics are retained. Enumeration
and search have coverage limits; truncated results are not a complete inventory.
Search uses the existing Python regular-expression semantics: the scan deadline
is cooperative, and a pathological individual regex match has no hard timeout.

`write_file(path, content, old_text=None, expected_hash=None)` creates/overwrites,
or uniquely replaces a nonempty exact old_text with content. Empty content is
valid; empty old_text and non-text content are not. Zero/multiple matches leave
the original untouched. No fuzzy matching, implicit replace-all, or mandatory
pre-read is added. An optional expected_hash compares complete original bytes;
a missing target with a supplied hash conflicts. Streaming edits use bounded
chunks and same-filesystem staging, still requiring O(N) IO and temporary disk.

## Tool selection and behavior contracts

* Workspace tools use behavior revision 1 for the current unpublished contract.
* `attach_files(paths=[...])` becomes `read_file(path=[...])`; explicit whole-text
  attachment becomes `read_file(path=..., as_file=True)`.
* `find_files("*.png")` becomes `list_files(include_glob="**/*.png")` for its old
  recursive basename intent. Old search `*.py` recursive intent likewise becomes
  `**/*.py`; directory prefixes must be rebased to the supplied path.
* Workspace host `read_file`, `search_files`, `find_files`, `attach_files` tool and
  `edit_file` remain available under their current contracts. New bounded
  host APIs are `read_text`, `inspect_file`, `list_files`, and `search_text`.
* Captured `attach_files` source/IDs and original messages are never rewritten.

## Extension example

`examples/skills/email-preview` converts a previously unsupported RFC 822 email
body through an explicit, bounded Python CLI. Its text and provenance output are
read by the same tools; no core email parser is added. Deploy the Skill/script
through the existing trusted Asset/Skill mechanism. The converter documents lost
headers, attachments and HTML layout; missing dependencies, malformed inputs,
permissions and timeout are real failures, not invitations to download tools.

Install `linktools-ai[magic]` and the platform's libmagic plus database through
normal deployment. No runtime tool installs these dependencies. Test the target
image: a Python binding alone does not prove the native database is available.
