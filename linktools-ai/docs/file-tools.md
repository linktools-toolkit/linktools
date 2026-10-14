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
legacy host entry points remain available when deliberately selected.
An edit-only configuration keeps `edit_file`; it is not upgraded to whole-file
write permission. Broad `*` selectors and the implicit compatibility default Agent
remain broad (the full 14-tool catalog), not a minimal profile. The minimal writer
example explicitly selects the four modern file tools.

The currently implemented provider adapter is OpenAI Chat (including compatible
endpoints). These explicit declarations describe that adapter; they do not claim
that other providers have been implemented. Custom legacy ModelBinding objects
retain their existing compatibility behavior. A future provider owns its own
encoding decisions at the model-binding boundary, not in file or business tools.

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

New explicit model input declarations default unknown types to rejection.
A compatible route may explicitly choose passthrough; it remains unverified and
cannot override SDK encoding, byte limits, Sandbox authorization, or permissions.
New configurations use input capabilities alone; do not also set `vision`.
Supplying both is rejected rather than creating conflicting policies. Omitted
input declarations preserve existing vision-only route contracts and recovery:
legacy `vision=false` forbids images, while old non-image inputs retain their
behavior. Legacy support is interpreted only at the compatibility boundary;
`vision=true` is not verified support for every image MIME.

### Minimal model declaration

```python
from linktools.ai.model import ModelInputCapability, ModelRegistry

models = ModelRegistry.openai(
    model="your-verified-model",
    input_capabilities=(
        ModelInputCapability("image/png"),
        ModelInputCapability("application/pdf", max_bytes=8 * 1024 * 1024),
        ModelInputCapability("text/csv"),
    ),
)
```

The host is responsible for verifying that this provider/model accepts the listed
MIME types. These declarations are examples, not claims about a named remote
model. Exact MIME values avoid a wildcard such as image/* falsely asserting every
codec. Unsupported declarations override passthrough for that MIME. The adapter
infers transport and captures the resolved meaning; equivalent explicit/default
transport declarations have the same contract. An advanced route that separately
supports uploaded CSV file blocks may additionally declare
`ModelInputCapability("text/csv", transport="file-block")`. The ordinary CSV
statement only verifies its normal inline transport; it does not silently verify
a different uploaded-file transport. `unknown_inputs="passthrough"` is explicit,
unverified compatibility behavior, not a substitute for known support.

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

The unified `read_file` uses `mime_type` in its `files`/`texts` result entries,
retaining `input_index` and `sha256` where applicable. Legacy `attach_files` keeps
its existing `media_type` contract, and old stored read results remain readable.
SDK content and internal attachment facts retain their own `media_type` field.
Runtime binding adds the existing attachment identifier to each new `read_file`
file entry. These tool results already
carry the metadata, so the SDK-added user content contains only the binary, not a
second copy of the metadata text.

After a complete response, active context retains the existing metadata and
removes only the binary content blocks, omitting the resulting empty user content
block while preserving tool call/return pairing. No consumed-content prefix is
added. Old stored labels and markers remain readable without rewriting raw history. Legacy
binaries lacking a reliable metadata association receive one metadata text block
when projected. URL inputs and uploaded-file references retain their existing
lifetime; this policy applies only to BinaryContent.

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

## Migration and compatibility

* Model-visible read_file/write_file/search_files change named behavior revision
  from 1 to 2. New list_files starts at 1; unchanged tools and wire versions do not
  change. Capture/recompile affected Agent bindings deliberately.
* `attach_files(paths=[...])` becomes `read_file(path=[...])`; explicit whole-text
  attachment becomes `read_file(path=..., as_file=True)`.
* `find_files("*.png")` becomes `list_files(include_glob="**/*.png")` for its old
  recursive basename intent. Old search `*.py` recursive intent likewise becomes
  `**/*.py`; directory prefixes must be rebased to the supplied path.
* Sandbox host `read_file`, `search_files`, `find_files`, `attach_files` tool and
  legacy edit_file remain available under their committed contracts. New bounded
  host APIs are `read_text`, `inspect_file`, `list_files`, and `search_text`.
* Historical attach_files source/IDs and original messages are never rewritten.

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
