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
