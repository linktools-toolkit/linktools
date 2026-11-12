#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution bindings and immutable payload bodies in the shared content archive."""

import hashlib
import json
from collections import Counter
from weakref import WeakKeyDictionary, ref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload
from ..execution import ExecutionRecord, ExecutionTimelineRecord, PendingToolContinuation, RecoveryTerminalHandoff, ResultRecord
from ..model import StoredUserInput
from ..agent import AgentBindingContract
from ..task import TaskBindingContract
from ._archive import archive_content_scope, archive_content_identity, ContentArchive, ContentRecord, PreparedArchive
from ._payload import decode_payload_bytes
from ._message_root import NativeTextReference, NativeUserInputReference, NativeMessageRoot
from ..steps import native_store_key
from ._step_store import SqlStepStore
from ..model.messages import decode_model_messages
from pydantic_ai.messages import ModelRequest, UserPromptPart, ModelResponse, TextPart
from .codec import (decode_execution_terminal_details,  binding_reference, decode_envelope, encode_envelope, _encode_persisted_domain,
    encode_execution_content, decode_execution_content, decode_execution_payload, execution_payload_reference, decode_domain, execution_payload_slots, payload_reference, native_user_input_reference, validate_native_user_input, native_text_reference, validate_native_text_payload, map_execution_payload_slots)
from .state import GroupRecordReader, StateStorageGroup, StateStore, StateTransaction, StoredRecord, active_state_scope

from ._execution_policy_content import (POLICY_CODEC, policy_reference, policy_values, encode_policy, decode_policy, expand_policy)

BINDING_CODEC = "execution-binding"
PAYLOAD_CODEC = "execution-payload"


class _ContentReadRequired(AIError):
    """A declared Execution reference needs an unopened snapshot member."""

    def __init__(self, group: StateStorageGroup) -> None:
        super().__init__(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self.group = group


def _payloads(record: StoredRecord) -> tuple[tuple[tuple[str, ...], Mapping], ...]:
    return execution_payload_slots(decode_envelope(record.data).value["payload"])


def _map_payloads(record: StoredRecord, mapper: Callable[[Mapping], JsonValue]) -> StoredRecord:
    envelope = decode_envelope(record.data)
    return replace(record, data=encode_envelope({**envelope.value,
        "payload": map_execution_payload_slots(envelope.value["payload"], mapper)}))


def _content(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord]) -> Iterator[tuple[tuple[str, str, str], ContentRecord]]:
    if record.kind != "execution":
        return
    references = [(binding_reference(_binding(record)), BINDING_CODEC), (policy_reference(record), POLICY_CODEC)]
    references.extend((payload_reference(payload), PAYLOAD_CODEC) for _, payload in _payloads(record))
    for reference, codec in references:
        if reference is None:
            continue
        content = contents.get(reference)
        if (content is None or content.identity != reference[2] or content.codec != codec
                or content.payload is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        yield reference, content


def _decode_payload(raw: bytes, identity: str) -> StoredPayload | StoredUserInput | PendingToolContinuation | RecoveryTerminalHandoff | ResultRecord | Mapping:
    if archive_content_identity(PAYLOAD_CODEC, hashlib.sha256(raw).hexdigest()) != identity:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        target = {"stored_user_input": StoredUserInput, "stored_payload": StoredPayload,
                  "pending_tool_continuation": PendingToolContinuation,
                  "recovery_terminal_handoff": RecoveryTerminalHandoff, "result_record": ResultRecord}.get(value.get("$dataclass"))
        if target is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value if target in {PendingToolContinuation, RecoveryTerminalHandoff, ResultRecord} else decode_domain(value, target)
    except (UnicodeError, ValueError, TypeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error



def _binding(record: StoredRecord) -> object:
    try:
        payload = decode_envelope(record.data).value["payload"]
        if not isinstance(payload, Mapping) or payload.get("$dataclass") != "execution_record":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        fields = payload["fields"]
        if not isinstance(fields, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        binding = fields["binding"]
        if not isinstance(binding, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return binding
    except (KeyError, TypeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _with_binding(record: StoredRecord, binding: JsonValue) -> StoredRecord:
    envelope = decode_envelope(record.data)
    payload = envelope.value["payload"]
    return replace(record, data=encode_envelope({**envelope.value, "payload": {
        **payload, "fields": {**payload["fields"], "binding": binding},
    }}))


def expand_binding(record: StoredRecord, raw: bytes) -> StoredRecord:
    """Restore the original wire bytes only after checking their content identity."""
    reference = binding_reference(_binding(record))
    if reference is None or archive_content_identity(BINDING_CODEC, hashlib.sha256(raw).hexdigest()) != reference[2]:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        binding = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(binding, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    decode_domain(binding, AgentBindingContract | TaskBindingContract)
    return _with_binding(record, binding)


def snapshot_execution_content_size(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord]) -> int:
    """Return decoded bytes retained for this execution, without hydrating them."""
    return sum(content.raw_size for _reference, content in _content(record, contents))


def expand_snapshot_execution_content(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord],
                                     *, namespace: str, tenant_id: str,
                                     reserve: Callable[[int], None] | None = None) -> StoredRecord:
    return _expand_execution_content(record, contents, content_scope=archive_content_scope(namespace, tenant_id),
                                     reserve=reserve)


def expand_retained_execution_content(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord]) -> StoredRecord:
    """Resolve retained owners across all scopes in one offline physical store."""
    return _expand_execution_content(record, contents, content_scope=None)


def _expand_execution_content(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord],
                              *, content_scope: str | None,
                              reserve: Callable[[int], None] | None = None) -> StoredRecord:
    if record.kind != "execution":
        return record
    decoded = {}
    for reference, content in _content(record, contents):
        if content_scope is not None and reference[1] != content_scope:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if reserve is not None:
            reserve(content.raw_size)
        decoded[reference] = decode_payload_bytes(content.body, codec=content.compression,
            raw_digest=content.raw_digest, raw_size=content.raw_size)
    policy = policy_reference(record)
    if policy is not None:
        record = expand_policy(record, decoded[policy])
    binding = binding_reference(_binding(record))
    if binding is not None:
        record = expand_binding(record, decoded[binding])
    values = {reference: _decode_payload(raw, reference[2]) for reference, raw in decoded.items()
              if reference != binding and reference != policy}
    def expand(payload: Mapping) -> JsonValue:
        reference = payload_reference(payload)
        return payload if reference is None else _encode_persisted_domain(decode_execution_payload(payload, values))
    return _map_payloads(record, expand)


def restore_native_text_payload(reference: NativeTextReference, raw: bytes) -> StoredPayload:
    """Rebuild and authenticate the exact JSON result from a fixed message."""
    try:
        messages = decode_model_messages(raw)
        if (len(messages) != 1 or not isinstance(messages[0], ModelResponse)
                or len(messages[0].parts) != 1 or not isinstance(messages[0].parts[0], TextPart)):
            raise ValueError("native result source is not one response text")
        payload = StoredPayload.inline_json(messages[0].parts[0].content)
        return payload
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def restore_native_user_input(reference: NativeUserInputReference, raw: bytes) -> StoredUserInput:
    """Authenticate the original plain-text accepted input from a fixed prompt."""
    try:
        messages = decode_model_messages(raw)
        if (len(messages) != 1 or not isinstance(messages[0], ModelRequest)
                or reference.part_index >= len(messages[0].parts)):
            raise ValueError("native input source is not a request part")
        part = messages[0].parts[reference.part_index]
        if not isinstance(part, UserPromptPart) or not isinstance(part.content, str):
            raise ValueError("native input source is not ordinary prompt text")
        return validate_native_user_input(reference, StoredUserInput("text", StoredPayload.inline_text(part.content)))
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def expand_native_execution_content(
    record: StoredRecord, payloads: Mapping[NativeTextReference | NativeUserInputReference, StoredPayload | StoredUserInput | PendingToolContinuation | RecoveryTerminalHandoff | ResultRecord | Mapping],
) -> StoredRecord:
    if record.kind != "execution":
        return record
    def expand(payload: Mapping) -> JsonValue:
        reference = native_user_input_reference(payload) or native_text_reference(payload)
        if reference is None:
            return payload
        restored = payloads.get(reference)
        return _encode_persisted_domain(validate_native_user_input(reference, restored) if isinstance(reference, NativeUserInputReference)
                                        else validate_native_text_payload(reference, restored))
    return _map_payloads(record, expand)


class ExecutionContent:
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        self._store, self._namespace, self._tenant_id = store, namespace, tenant_id
        self._content_archive: ContentArchive | None = None
        self._share_content = True
        # This cache saves immutable encoding work, never proves source existence.
        self._payload_encodings: dict[int, tuple[ref, str, bytes]] = {}
        self._binding_encodings: dict[int, tuple[ref, str, bytes]] = {}
        self._projections: dict[int, tuple[ref, StoredRecord]] = {}
        self._binding_reads: WeakKeyDictionary[StateTransaction, dict[str, tuple[ContentRecord, bytes]]] = WeakKeyDictionary()

    def enable_sharing(self) -> None:
        """Allow immutable Execution content to be prepared at its next write boundary."""
        self._share_content = True

    async def initialize(self, *, create: bool, share_content: bool = True) -> None:
        if share_content:
            self.enable_sharing()
        if self._content_archive is not None:
            return
        group = self._store.storage_group
        if not isinstance(group, StateStorageGroup):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        active = active_state_scope()
        if active is not None:
            opened = group.opened_archive
            if opened is None:
                return
            if active.group is not group or opened[1] not in active.transactions:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            opened = group.opened_archive or await group.open_archive(create=create)
        if opened is None:
            return
        scope, store = opened
        # Immutable Execution bodies stay in the database for atomic publication
        # and complete offline snapshot validation before state is admitted.
        self._content_archive = ContentArchive(store, scope, namespace=self._namespace,
                                               tenant_id=self._tenant_id, object_store=None)

    async def _archive(self, *, create: bool = False) -> ContentArchive:
        active = active_state_scope()
        group = self._store.storage_group
        if not create and active is not None and not active.writable and active.group is group:
            opened = group.opened_archive
            if opened is None or opened[1] not in active.transactions:
                raise _ContentReadRequired(group)
        await self.initialize(create=create, share_content=False)
        if self._content_archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        active = active_state_scope()
        if active is not None and (active.group is not self._store.storage_group
                                   or self._content_archive.store not in active.transactions):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._content_archive

    async def prefetch_configuration_owner(self, hint: StoredRecord) -> None:
        """Use earlier reference addresses only to batch a fresh owner observation."""
        active = active_state_scope()
        if (active is None or active.writable or active.group is not self._store.storage_group
                or not isinstance(active.record_reader, GroupRecordReader)):
            return
        try:
            references = tuple(reference for reference in (binding_reference(_binding(hint)), policy_reference(hint))
                               if reference is not None)
        except AIError:
            return
        if not references:
            return
        archive = await self._archive()
        if any(reference[:2] != (archive.scope_id, archive.content_scope) for reference in references):
            return
        # These are hints, not retained values or source-existence evidence. The
        # current owner and hinted content are both read in the active snapshot;
        # normal decoding follows the current owner's refs, including changes.
        await active.record_reader.get_records({
            self._store: (hint.key,),
            archive.store: archive.content_identity_keys(tuple(reference[2] for reference in references)),
        })

    async def expand(self, record: StoredRecord) -> StoredRecord:
        return (await self.expand_many((record,)))[0]

    async def expand_policies(self, records: Sequence[StoredRecord], *,
                              reserve: Callable[[int], None] | None = None) -> tuple[StoredRecord, ...]:
        return await self._expand_configuration(records, bindings=False, policies=True, reserve=reserve)

    async def expand_bindings(self, records: Sequence[StoredRecord]) -> tuple[StoredRecord, ...]:
        return await self._expand_configuration(records, bindings=True, policies=True)

    async def _expand_configuration(self, records: Sequence[StoredRecord], *, bindings: bool = True,
                                    policies: bool = False,
                                    reserve: Callable[[int], None] | None = None) -> tuple[StoredRecord, ...]:
        binding_refs = tuple(binding_reference(_binding(record)) if bindings else None for record in records)
        policy_refs = tuple(policy_reference(record) if policies else None for record in records)
        references = (*binding_refs, *policy_refs)
        if not any(reference is not None for reference in references):
            return tuple(records)
        archive = await self._archive()
        if any(reference is not None and reference[:2] != (archive.scope_id, archive.content_scope)
               for reference in references):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        counts = Counter(reference[2] for reference in references if reference is not None)
        identities = tuple(counts)
        codecs = {reference[2]: codec for refs, codec in ((binding_refs, BINDING_CODEC), (policy_refs, POLICY_CODEC))
                  for reference in refs if reference is not None}
        # Both configuration families belong to the same owner snapshot. Capture
        # their metadata together, rather than querying each family and then
        # querying binding metadata again just to load its inline bytes.
        captured, load = await archive.capture_content_records(identities)
        records_by_id = dict(zip(identities, captured, strict=True))
        active = active_state_scope()
        transaction = None if active is None else active.transactions.get(archive.store)
        cached = None
        if transaction is not None:
            try:
                cached = self._binding_reads.setdefault(transaction, {})
            except TypeError:
                pass
        contents, missing = {}, []
        for identity, content in records_by_id.items():
            if content.codec != codecs[identity] or content.payload is not None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if reserve is not None:
                reserve(counts[identity] * content.raw_size)
            previous = None if cached is None else cached.get(identity)
            if previous is not None and previous[0] == content:
                contents[identity] = previous[1]
            else:
                missing.append(identity)
        raws = await load(missing)
        for identity, raw in zip(missing, raws, strict=True):
            contents[identity] = raw
            if cached is not None:
                cached[identity] = records_by_id[identity], raw
        expanded = []
        for record, binding, policy in zip(records, binding_refs, policy_refs, strict=True):
            if policy is not None:
                record = expand_policy(record, contents[policy[2]])
            if binding is not None:
                record = expand_binding(record, contents[binding[2]])
            expanded.append(record)
        return tuple(expanded)


    def remember_projection(self, value: ExecutionRecord, record: StoredRecord) -> None:
        key = id(value)
        existing = self._projections.get(key)
        if existing is None or existing[0]() is not value:
            # A later uncommitted representation must not replace this value's
            # original source projection. New control values get their own entry.
            self._projections[key] = (ref(value, lambda _value: self._projections.pop(key, None)), record)

    def projection(self, value: ExecutionRecord) -> StoredRecord | None:
        cached = self._projections.get(id(value))
        return cached[1] if cached is not None and cached[0]() is value else None

    @staticmethod
    def payload_values(value: ExecutionRecord) -> tuple[StoredPayload | StoredUserInput | PendingToolContinuation | RecoveryTerminalHandoff | ResultRecord | Mapping, ...]:
        values = [value.stored_user_input]
        if value.result is not None and value.result.output is not None:
            values.append(value.result.output)
        recovery = value.recovery
        handoff = None if recovery is None else recovery.terminal_handoff
        if recovery is not None and recovery.pending_tools is not None:
            values.extend(call.arguments_payload for call in (*recovery.pending_tools.approvals, *recovery.pending_tools.calls))
            values.append(recovery.pending_tools)
        if handoff is not None and handoff.outcome.output is not None:
            values.append(handoff.outcome.output)
        if value.result is not None:
            values.append(value.result)
        if handoff is not None:
            values.append(handoff)
        return tuple(values)

    def _encoding(self, payload: StoredPayload | StoredUserInput | PendingToolContinuation | RecoveryTerminalHandoff | ResultRecord | Mapping) -> tuple[str, bytes]:
        key = id(payload)
        cached = self._payload_encodings.get(key)
        if cached is not None and cached[0]() is payload:
            return cached[1:]
        from .codec import encode_domain
        wire = (_encode_persisted_domain(payload) if isinstance(payload, (PendingToolContinuation, ResultRecord, RecoveryTerminalHandoff))
                else encode_domain(payload))
        if isinstance(payload, ResultRecord):
            wire = {**wire, "fields": {**wire["fields"], "output": None}}
        elif isinstance(payload, PendingToolContinuation):
            fields_value = dict(wire["fields"])
            for name in ("approvals", "calls"):
                fields_value[name] = {"$tuple": [{**call, "fields": {**call["fields"], "arguments_payload": None}}
                    for call in fields_value[name]["$tuple"]]}
            wire = {**wire, "fields": fields_value}
        elif isinstance(payload, RecoveryTerminalHandoff):
            outcome = wire["fields"]["outcome"]
            outcome_fields = {name: item for name, item in outcome["fields"].items()
                              if name not in {"safe_error_details", "error_diagnostics"}}
            outcome = {**outcome, "fields": {**outcome_fields, "output": None}}
            wire = {**wire, "fields": {**wire["fields"], "outcome": outcome}}
        raw = canonical_json_bytes(wire)
        identity = archive_content_identity(PAYLOAD_CODEC, hashlib.sha256(raw).hexdigest())
        try:
            weak = ref(payload, lambda _value: self._payload_encodings.pop(key, None))
        except TypeError:
            # Slotted result/control values need not expose a weakref slot.
            # Encoding caches are optional and must not retain their owners.
            return identity, raw
        self._payload_encodings[key] = (weak, identity, raw)
        return identity, raw

    def _binding_encoding(self, binding: AgentBindingContract | TaskBindingContract) -> tuple[str, bytes]:
        key = id(binding)
        cached = self._binding_encodings.get(key)
        if cached is not None and cached[0]() is binding:
            return cached[1:]
        raw = canonical_json_bytes(_encode_persisted_domain(binding))
        identity = archive_content_identity(BINDING_CODEC, hashlib.sha256(raw).hexdigest())
        self._binding_encodings[key] = (ref(binding, lambda _value: self._binding_encodings.pop(key, None)), identity, raw)
        return identity, raw

    async def prepare_value(self, value: ExecutionRecord) -> None:
        if self._share_content:
            await self.initialize(create=True)

    async def prepare_terminal_content(
        self, value: ExecutionRecord, prepared: PreparedArchive, *, result_source: NativeTextReference | None = None,
    ) -> None:
        """Add newly terminal bytes to the caller's matching publication bundle."""
        if value.result is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        selected = [value.result]
        if result_source is None and value.result.output is not None:
            selected.append(value.result.output)
        if value.safe_error_details or value.error_diagnostics is not None:
            selected.append(StoredPayload.inline_json({
                "safe_error_details": _encode_persisted_domain(value.safe_error_details),
                "error_diagnostics": _encode_persisted_domain(value.error_diagnostics),
            }))
        archive = await self._archive(create=True)
        encoded = tuple(self._encoding(payload) for payload in selected)
        await archive.prepare_content(tuple((PAYLOAD_CODEC, raw) for identity, raw in encoded), prepared)

    async def encode(self, value: ExecutionRecord, *, previous: StoredRecord | None = None,
                     result_source: NativeTextReference | None = None,
                     observed_value: ExecutionRecord | None = None,
                     verified_native_previous: ExecutionRecord | None = None) -> JsonValue:
        """Publish new immutable bodies with the same transaction as their owner."""
        if observed_value is not None and (previous is None or self.projection(observed_value) is not previous):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        output = None if value.result is None else value.result.output
        # Native locations own the value, not a redundant digest in records.
        # Reuse requires comparing the current canonical source, never trusting
        # a proposed in-memory value merely because its type still matches.
        if verified_native_previous is not None:
            # Only the terminal writer supplies this proof: it decoded this
            # exact projection in its current transaction, and has performed
            # no source writes since. Other paths must authenticate anew.
            active = active_state_scope()
            if (previous is None or self.projection(verified_native_previous) is not previous
                    or active is None or not active.writable
                    or self._store not in active.transactions):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous_native = {}
            for path, payload in _payloads(previous):
                reference = native_user_input_reference(payload) or native_text_reference(payload)
                if reference is None:
                    continue
                if path == ("stored_user_input",):
                    previous_native[reference] = verified_native_previous.stored_user_input
                elif path == ("result", "fields", "output") and verified_native_previous.result is not None:
                    previous_native[reference] = verified_native_previous.result.output
                elif (path == ("recovery", "fields", "terminal_handoff", "fields", "output")
                      and verified_native_previous.recovery is not None
                      and verified_native_previous.recovery.terminal_handoff is not None):
                    previous_native[reference] = verified_native_previous.recovery.terminal_handoff.outcome.output
                else:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            previous_native = (await self.read_payloads((previous,)) if previous is not None and any(
                native_user_input_reference(payload) is not None or native_text_reference(payload) is not None
                for _, payload in _payloads(previous)) else {})
        if result_source is None and previous is not None and output is not None:
            for path, payload in _payloads(previous):
                if path == ("result", "fields", "output"):
                    previous_source = native_text_reference(payload)
                    if previous_source is not None and previous_native.get(previous_source) == output:
                        result_source = previous_source
        if result_source is not None:
            opened = self._store.storage_group.opened_archive
            if (output is None or output.kind != "inline" or output.encoding != "json"
                    or not isinstance(output.decode(), str)
                    or opened is None or result_source.root.group_id != opened[0]
                    or result_source.root.store_key != native_store_key(self._namespace, self._tenant_id)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        input_source = None
        if previous is not None:
            for path, payload in _payloads(previous):
                if path == ("stored_user_input",):
                    source = native_user_input_reference(payload)
                    if source is not None:
                        validate_native_user_input(source, value.stored_user_input)
                        if previous_native.get(source) != value.stored_user_input:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        input_source = source
        terminal_details = (None if not value.safe_error_details and value.error_diagnostics is None else
            StoredPayload.inline_json({
                "safe_error_details": _encode_persisted_domain(value.safe_error_details),
                "error_diagnostics": _encode_persisted_domain(value.error_diagnostics),
            }))
        handoff = None if value.recovery is None else value.recovery.terminal_handoff
        handoff_details = (None if handoff is None or not handoff.outcome.safe_error_details
            and handoff.outcome.error_diagnostics is None else StoredPayload.inline_json({
                "safe_error_details": _encode_persisted_domain(handoff.outcome.safe_error_details),
                "error_diagnostics": _encode_persisted_domain(handoff.outcome.error_diagnostics),
            }))
        details_values = tuple(item for item in (terminal_details, handoff_details) if item is not None)
        selected = [(payload, self._encoding(payload)) for payload in (*details_values, *self.payload_values(value))
                    if (result_source is None or payload is not output)
                    and (input_source is None or payload is not value.stored_user_input)]
        selected = [(payload, encoded) for payload, encoded in selected if encoded is not None]
        binding = self._binding_encoding(value.binding)
        active = active_state_scope()
        opened = self._store.storage_group.opened_archive
        if active is not None and (opened is None or opened[1] not in active.transactions):
            # An accepted execution must never fall back to a second inline owner.
            # Callers entering their own transaction must prepare content first.
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        archive = await self._archive(create=True)
        retained = {reference for _, payload in _payloads(previous)
                    if (reference := payload_reference(payload)) is not None} if previous is not None else set()
        if any(reference[:2] != (archive.scope_id, archive.content_scope) for reference in retained):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        previous_binding = None if previous is None else binding_reference(_binding(previous))
        if previous_binding is not None and previous_binding[:2] != (archive.scope_id, archive.content_scope):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        encoded_payloads = {} if result_source is None else {id(output): {
            "$dataclass": "stored_payload", "schema": 1, "fields": {"content": result_source.to_json()}}}
        if input_source is not None:
            encoded_payloads[id(value.stored_user_input)] = {"$dataclass": "stored_user_input", "schema": 1,
                                                           "fields": {"content": input_source.to_json()}}
        missing = []
        for payload, (identity, raw) in selected:
            reference = archive.scope_id, archive.content_scope, identity
            if reference not in retained:
                missing.append((PAYLOAD_CODEC, raw))
            wire = {StoredUserInput: "stored_user_input", StoredPayload: "stored_payload",
                    PendingToolContinuation: "pending_tool_continuation", RecoveryTerminalHandoff: "recovery_terminal_handoff", ResultRecord: "result_record"}[type(payload)]
            encoded_payloads[id(payload)] = {"$dataclass": wire, "schema": 1,
                "fields": {"content": {"$execution_payload": identity,
                    "archive_scope": archive.scope_id, "content_scope": archive.content_scope}}}
            if isinstance(payload, PendingToolContinuation):
                encoded_payloads[id(payload)]["fields"]["arguments"] = {
                    str(index): encoded_payloads[id(call.arguments_payload)]
                    for index, call in enumerate((*payload.approvals, *payload.calls))}
            if isinstance(payload, RecoveryTerminalHandoff):
                encoded_payloads[id(payload)]["fields"]["terminal_details"] = (
                    None if handoff_details is None else encoded_payloads[id(handoff_details)])
            if isinstance(payload, (ResultRecord, RecoveryTerminalHandoff)):
                body = payload.output if isinstance(payload, ResultRecord) else payload.outcome.output
                encoded_payloads[id(payload)]["fields"]["output"] = None if body is None else encoded_payloads[id(body)]
        identity, raw = binding
        if previous_binding != (archive.scope_id, archive.content_scope, identity):
            missing.append((BINDING_CODEC, raw))
        previous_policy = None if previous is None else policy_reference(previous)
        effective_policy = policy_values(value)
        if previous_policy is None:
            policy_identity, policy_raw = encode_policy(effective_policy)
            missing.append((POLICY_CODEC, policy_raw))
            policy_overrides = {}
        else:
            if previous_policy[:2] != (archive.scope_id, archive.content_scope):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            policy_identity = previous_policy[2]
            if (observed_value is not None
                    and canonical_json_bytes(effective_policy) == canonical_json_bytes(policy_values(observed_value))):
                # A confirmed observed boundary retains its unchanged fixed policy.
                # The caller still CASes that owner's physical revision; ordinary
                # reads and policy changes must inspect their current source.
                policy_overrides = decode_envelope(previous.data).value["payload"]["fields"].get("runtime_overrides", {})
            else:
                policy_raw, = await archive.read_contents((previous_policy[2],), codec=POLICY_CODEC)
                base_policy = decode_policy(previous_policy, policy_raw)
                policy_overrides = {name: item for name, item in effective_policy.items()
                                    if canonical_json_bytes(item) != canonical_json_bytes(base_policy[name])}
        encoded_policy = {"$runtime_policy": policy_identity, "archive_scope": archive.scope_id,
                          "content_scope": archive.content_scope}
        encoded_binding = {"$execution_binding": identity, "archive_scope": archive.scope_id,
                           "content_scope": archive.content_scope}
        if missing:
            await archive.ensure_content(tuple(missing))
        return encode_execution_content(value, encoded_payloads, binding=encoded_binding,
                                        policy=encoded_policy, policy_overrides=policy_overrides,
                                        terminal_details=None if terminal_details is None else encoded_payloads[id(terminal_details)])

    @staticmethod
    def has_native_input_candidate(prepared: PreparedArchive) -> bool:
        """Inspect new published bodies for an exact input source at any size."""
        for sequence in prepared.native_messages.values():
            for _identity, content in sequence.messages:
                if content.payload is not None:
                    continue
                raw = decode_payload_bytes(content.body, codec=content.compression,
                                           raw_digest=content.raw_digest, raw_size=content.raw_size)
                for message in decode_model_messages(raw):
                    if isinstance(message, ModelRequest) and any(isinstance(part, UserPromptPart)
                            and isinstance(part.content, str) for part in message.parts):
                        return True
        return False

    async def project_user_input(self, record: StoredRecord, roots: Sequence[NativeMessageRoot]) -> StoredRecord | None:
        """Authenticate published source bytes before replacing an accepted input."""
        wrapper = next(payload for path, payload in _payloads(record) if path == ("stored_user_input",))
        if native_user_input_reference(wrapper) is not None:
            return None
        reference = payload_reference(wrapper)
        # Additive wrapper metadata remains legal to read, but must not be lost
        # when choosing a different storage representation.
        if set(wrapper) != {"$dataclass", "schema", "fields"}:
            return None
        archive = await self._archive()
        if reference is None:
            original = decode_domain(wrapper, StoredUserInput)
            if wrapper != _encode_persisted_domain(original):
                return None
        else:
            if reference[:2] != (archive.scope_id, archive.content_scope):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            raw, = await archive.read_contents((reference[2],), codec=PAYLOAD_CODEC)
            original = _decode_payload(raw, reference[2])
            from .codec import encode_domain
            if raw != canonical_json_bytes(encode_domain(original)):
                return None
        if (not isinstance(original, StoredUserInput) or original.codec != "text" or original.view is not None
                or original.payload.kind != "inline" or original.payload.encoding != "utf-8"):
            return None
        if any(root.group_id != archive.scope_id or root.store_key != native_store_key(self._namespace, self._tenant_id)
               for root in roots):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native = SqlStepStore.for_state_store(self._store, namespace=self._namespace, tenant_id=self._tenant_id)
        active = active_state_scope()
        transaction = None if active is None or not active.writable else active.transactions.get(self._store)
        if native is None or transaction is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native.bind_message_archive(archive)
        batches = await (await native.capture_message_roots(roots, transaction=transaction))()
        for root, messages in zip(roots, batches, strict=True):
            for message_index, raw in enumerate(messages):
                decoded = decode_model_messages(raw)
                if len(decoded) != 1 or not isinstance(decoded[0], ModelRequest):
                    continue
                for part_index, part in enumerate(decoded[0].parts):
                    if not isinstance(part, UserPromptPart) or part.content != original.payload.decode():
                        continue
                    source = NativeUserInputReference(root.slice(message_index, message_index + 1), part_index)
                    if restore_native_user_input(source, raw) != original:
                        continue
                    replacement = {"$dataclass": "stored_user_input", "schema": 1,
                                   "fields": {"content": source.to_json()}}
                    envelope = decode_envelope(record.data)
                    payload = envelope.value["payload"]
                    return replace(record, storage_revision=record.storage_revision + 1, data=encode_envelope({
                        **envelope.value, "payload": {**payload, "fields": {**payload["fields"], "stored_user_input": replacement}}}))
        return None

    async def verify_result_source(self, source: NativeTextReference, output: StoredPayload | None) -> None:
        """Authenticate stored source bytes inside the owner's write transaction."""
        validate_native_text_payload(source, output)
        archive = await self._archive()
        if (source.root.group_id != archive.scope_id
                or source.root.store_key != native_store_key(self._namespace, self._tenant_id)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        active = active_state_scope()
        native = SqlStepStore.for_state_store(self._store, namespace=self._namespace, tenant_id=self._tenant_id)
        transaction = None if active is None or not active.writable else active.transactions.get(self._store)
        if native is None or transaction is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native.bind_message_archive(archive)
        messages, = await (await native.capture_message_roots((source.root,), transaction=transaction))()
        if len(messages) != 1 or restore_native_text_payload(source, messages[0]) != output:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    async def read_payloads(self, records: Sequence[StoredRecord], *,
                            reserve: Callable[[int], None] | None = None, result_only: bool = False,
                            reserve_native: Callable[[int], None] | None = None) -> dict[tuple[str, str, str] | NativeTextReference | NativeUserInputReference, StoredPayload | StoredUserInput | PendingToolContinuation | RecoveryTerminalHandoff | ResultRecord | Mapping]:
        native_references = []
        for record in records:
            for path, payload in _payloads(record):
                if result_only and path not in {("result",), ("result", "fields", "output")}:
                    continue
                reference = native_user_input_reference(payload) or native_text_reference(payload)
                if reference is not None:
                    native_references.append(reference)
        native_values = {}
        if native_references:
            archive = await self._archive()
            references_unique = tuple(dict.fromkeys(native_references))
            if any(source.root.group_id != archive.scope_id or source.root.store_key != native_store_key(
                    self._namespace, self._tenant_id) for source in references_unique):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            native = SqlStepStore.for_state_store(self._store, namespace=self._namespace, tenant_id=self._tenant_id)
            active = active_state_scope()
            transaction = None if active is None else active.transactions.get(self._store)
            if native is None or transaction is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            native.bind_message_archive(archive)
            batches = await (await native.capture_message_roots(
                tuple(source.root for source in references_unique), transaction=transaction, reserve=reserve_native or reserve))()
            for source, raw_messages in zip(references_unique, batches, strict=True):
                if len(raw_messages) != 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                payload = (restore_native_user_input(source, raw_messages[0]) if isinstance(source, NativeUserInputReference)
                           else restore_native_text_payload(source, raw_messages[0]))
                native_values[source] = payload
        references = [reference for record in records for path, payload in _payloads(record)
                      if (not result_only or path in {("result",), ("result", "fields", "output")})
                      if (reference := payload_reference(payload)) is not None]
        if not references:
            return native_values
        archive = await self._archive()
        if any(reference[:2] != (archive.scope_id, archive.content_scope) for reference in references):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        counts = Counter(reference[2] for reference in references)
        identities = tuple(counts)
        def reserve_content(identity: str, size: int) -> None:
            if reserve is not None:
                reserve(counts[identity] * size)
        raws = await archive.read_contents(identities, codec=PAYLOAD_CODEC,
            reserve=reserve_content if reserve is not None else None)
        values = dict(native_values)
        for identity, raw in zip(identities, raws, strict=True):
            value = _decode_payload(raw, identity)
            if not isinstance(value, Mapping):
                key = id(value)
                self._payload_encodings[key] = (ref(value, lambda _value, key=key: self._payload_encodings.pop(key, None)), identity, raw)
            values[(archive.scope_id, archive.content_scope, identity)] = value
        return values

    async def terminal_matches(self, current: StoredRecord, candidate: StoredRecord, value: ExecutionRecord) -> bool:
        # Recovery templates may disappear or change phase when a prepared
        # handoff is published. Their identities are not terminal facts. Read
        # the existing owners instead of treating candidate identities as a
        # substitute for the previous execution's semantic projection.
        current, = await self.expand_policies((current,))
        payloads = await self.read_payloads((current,))
        previous = decode_execution_content(current.data, payloads, timeline=True)
        return previous.completed_at is None or previous.terminal_matches(value)

    async def expand_projection_components(
        self, records: Sequence[StoredRecord], *, result: bool = False,
        recovery: bool = False, terminal_details: bool = False,
    ) -> tuple[StoredRecord, ...]:
        """Read requested control templates only, retaining input/output locations."""
        paths = set()
        if result:
            paths.add(("result",))
        if recovery:
            paths.update({("recovery", "fields", "pending_tools"),
                          ("recovery", "fields", "terminal_handoff"),
                          ("recovery", "fields", "terminal_handoff", "fields", "terminal_details")})
        if terminal_details:
            paths.add(("terminal_details",))
        references = {reference for record in records for path, payload in _payloads(record)
                      if path in paths and (reference := payload_reference(payload)) is not None}
        decoded = {}
        if references:
            archive = await self._archive()
            if any(reference[:2] != (archive.scope_id, archive.content_scope) for reference in references):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            ordered = tuple(references)
            raws = await archive.read_contents(tuple(reference[2] for reference in ordered), codec=PAYLOAD_CODEC)
            decoded = {reference: _decode_payload(raw, reference[2]) for reference, raw in zip(ordered, raws, strict=True)}
        values = []
        for record in records:
            envelope = decode_envelope(record.data)
            payload = envelope.value["payload"]
            fields_value = dict(payload["fields"])
            if terminal_details and "terminal_details" in fields_value and fields_value["terminal_details"] is None:
                if "safe_error_details" in fields_value or "error_diagnostics" in fields_value:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                fields_value.pop("terminal_details")
                fields_value.update(decode_execution_terminal_details(None))
            for path, wrapper in _payloads(record):
                if path == ("recovery", "fields", "terminal_handoff", "fields", "terminal_details"):
                    continue
                if path not in paths or (reference := payload_reference(wrapper)) is None:
                    continue
                value = decoded[reference]
                if path == ("terminal_details",):
                    details = decode_execution_terminal_details(value)
                    if "safe_error_details" in fields_value or "error_diagnostics" in fields_value:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    fields_value.pop("terminal_details")
                    fields_value.update(details)
                    continue
                expanded = dict(value) if isinstance(value, Mapping) else _encode_persisted_domain(value)
                if wrapper["$dataclass"] == "pending_tool_continuation":
                    from .codec import _restore_pending_arguments
                    expanded = _restore_pending_arguments(expanded, wrapper["fields"]["arguments"])
                if wrapper["$dataclass"] in {"result_record", "recovery_terminal_handoff"}:
                    expanded_fields = dict(expanded["fields"])
                    if wrapper["$dataclass"] == "result_record":
                        expanded_fields["output"] = wrapper["fields"]["output"]
                    else:
                        outcome = expanded_fields["outcome"]
                        details_wrapper = wrapper["fields"]["terminal_details"]
                        details_reference = None if details_wrapper is None else payload_reference(details_wrapper)
                        if details_wrapper is not None and details_reference is None:
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                        details = decode_execution_terminal_details(
                            None if details_reference is None else decoded[details_reference])
                        expanded_fields["outcome"] = {**outcome, "fields": {**outcome["fields"], **details,
                            "output": wrapper["fields"]["output"]}}
                    expanded["fields"] = expanded_fields
                current = fields_value
                for name in path[:-1]:
                    current[name] = dict(current[name])
                    current = current[name]
                current[path[-1]] = expanded
            values.append(replace(record, data=encode_envelope({**envelope.value,
                "payload": {**payload, "fields": fields_value}})))
        return tuple(values)

    async def expand_terminal_details(self, records: Sequence[StoredRecord]) -> tuple[StoredRecord, ...]:
        return await self.expand_projection_components(records, terminal_details=True)

    async def decode(self, record: StoredRecord, *, timeline: bool = False) -> ExecutionRecord | ExecutionTimelineRecord:
        return (await self.decode_many((record,), timeline=timeline))[0]

    async def decode_many(self, records: Sequence[StoredRecord], *,
                          timeline: bool = False) -> tuple[ExecutionRecord | ExecutionTimelineRecord, ...]:
        payloads = await self.read_payloads(records)
        expanded = await self.expand_policies(records) if timeline else await self.expand_bindings(records)
        values = []
        for original, record in zip(records, expanded, strict=True):
            value = decode_execution_content(record.data, payloads, timeline=timeline)
            if not timeline:
                self.remember_projection(value, original)
            values.append(value)
        return tuple(values)

    async def expand_many(self, records: Sequence[StoredRecord], *,
                          reserve: Callable[[int], None] | None = None,
                          reserve_native: Callable[[int], None] | None = None) -> tuple[StoredRecord, ...]:
        payloads = await self.read_payloads(records, reserve=reserve, reserve_native=reserve_native)
        records = await self.expand_policies(records, reserve=reserve)
        records = await self._expand_configuration(records, reserve=reserve)
        def expand(payload: Mapping) -> JsonValue:
            reference = native_user_input_reference(payload) or native_text_reference(payload) or payload_reference(payload)
            return payload if reference is None else _encode_persisted_domain(decode_execution_payload(payload, payloads))
        return tuple(_map_payloads(record, expand) for record in records)


__all__ = ["restore_native_user_input", "restore_native_text_payload", "expand_native_execution_content", "ExecutionContent", "expand_snapshot_execution_content", "expand_retained_execution_content", "snapshot_execution_content_size",
           ]
