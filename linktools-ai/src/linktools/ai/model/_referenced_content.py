#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Descriptor-only input whose immutable bytes remain with their Asset owner."""

import json
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from typing import Literal

from pydantic_ai.messages import TextContent

from ..asset import AssetVersionRef
from ..core import ImmutableJsonMapping, JsonValue, Principal, validate_persistence_namespace
from ..spec import CapabilityRef


@dataclass(frozen=True, slots=True)
class ReferencedInputOrigin:
    """Original tool producer coordinates, not an authorization grant.

    Runtime projects these from the actual admitted invocation. Consumers keep
    their own principal and execution context; hosts reauthorize against the
    producer's original admission before reading. Unknown request coordinates
    remain absent when Trace did not capture them.
    """

    namespace: str
    principal: Principal
    execution_id: str
    agent_run_id: str
    agent_run_seq: int | None
    model_request_seq: int | None
    tool_call_id: str
    tool_name: str

    def __post_init__(self) -> None:
        validate_persistence_namespace(self.namespace)
        if not isinstance(self.principal, Principal):
            raise TypeError("origin principal must be Principal")
        for value in (self.execution_id, self.agent_run_id, self.tool_call_id, self.tool_name):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("origin producer identity is incomplete")
        for value in (self.agent_run_seq, self.model_request_seq):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("origin sequence must be a positive integer or None")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "namespace": self.namespace,
            "principal": {"principal_id": self.principal.principal_id,
                          "tenant_id": self.principal.tenant_id, "kind": self.principal.kind},
            "execution_id": self.execution_id, "agent_run_id": self.agent_run_id,
            "agent_run_seq": self.agent_run_seq, "model_request_seq": self.model_request_seq,
            "tool_call_id": self.tool_call_id, "tool_name": self.tool_name,
        }

    @classmethod
    def from_payload(cls, value: object) -> "ReferencedInputOrigin":
        if not isinstance(value, Mapping) or set(value) != {
            "namespace", "principal", "execution_id", "agent_run_id", "agent_run_seq",
            "model_request_seq", "tool_call_id", "tool_name",
        }:
            raise ValueError("referenced input origin is invalid")
        principal = value["principal"]
        if not isinstance(principal, Mapping) or set(principal) != {"principal_id", "tenant_id", "kind"}:
            raise ValueError("referenced input origin principal is invalid")
        return cls(namespace=value["namespace"], principal=Principal(**principal),
                   execution_id=value["execution_id"], agent_run_id=value["agent_run_id"],
                   agent_run_seq=value["agent_run_seq"], model_request_seq=value["model_request_seq"],
                   tool_call_id=value["tool_call_id"], tool_name=value["tool_name"])


class ReferencedContent(TextContent):
    """Deliver a recoverable Asset without copying its bytes into Runtime storage.

    The selected Agent's ReferencedInputsCapability authorizes and loads this
    exact version for each provider request. The host must retain that version;
    a URL alone, or a mutable path resolved to its latest value, is insufficient.
    Runtime-owned codecs preserve this type; SDK JSON is not its durable codec.
    Descriptor attributes are not dataclass fields: SDK serialization exposes only
    the inherited text summary, including before provider hydration.
    Binary representation keeps file-input admission and encoding. Explicit text
    representation decodes verified UTF-8 bytes into an ordinary provider string;
    MIME metadata alone never chooses or changes the representation.
    Optional source_parameters carry immutable JSON reconstruction arguments for
    the selected loader, not content bodies, credentials or authorization proof.
    They are retained with the descriptor but omitted from its model-visible summary.
    An optional identifier associates an input occurrence with Runtime attachment
    facts. It is not Asset identity, a reconstruction argument or authorization.
    """

    origin: ReferencedInputOrigin | None
    ref: AssetVersionRef
    media_type: str
    source: CapabilityRef
    representation: "Literal['binary', 'text']"
    identifier: str | None
    source_parameters: Mapping[str, object]

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return (self.ref, self.media_type, self.source, self.representation,
                self.identifier, self.source_parameters, self.content, self.metadata, self.origin) == (
                    other.ref, other.media_type, other.source, other.representation,
                    other.identifier, other.source_parameters, other.content, other.metadata, other.origin)

    def __init__(
        self, ref: AssetVersionRef, *, media_type: str, source: CapabilityRef,
        representation: "Literal['binary', 'text']" = "binary",
        source_parameters: "Mapping[str, JsonValue] | None" = None,
        identifier: str | None = None,
        origin: ReferencedInputOrigin | None = None,
    ) -> None:
        if not isinstance(ref, AssetVersionRef):
            raise TypeError("referenced content requires an AssetVersionRef")
        if not isinstance(media_type, str) or not media_type.strip():
            raise ValueError("referenced content media_type must be non-empty")
        if (not isinstance(source, CapabilityRef) or source.kind != "capability"
            or not isinstance(source.id, str) or not source.id.strip()
            or type(source.revision) is not int or source.revision < 1):
            raise ValueError("referenced content source must name a capability revision")
        if representation not in ("binary", "text"):
            raise ValueError("referenced content representation must be binary or text")
        if source_parameters is not None and not isinstance(source_parameters, Mapping):
            raise TypeError("referenced content source_parameters must be a JSON mapping")
        if identifier is not None and not isinstance(identifier, str):
            raise TypeError("referenced content identifier must be a string")
        if origin is not None and not isinstance(origin, ReferencedInputOrigin):
            raise TypeError("referenced content origin must be ReferencedInputOrigin")
        self.origin = origin
        self.source_parameters = ImmutableJsonMapping({} if source_parameters is None else source_parameters)
        self.ref = ref
        self.media_type = media_type
        self.source = source
        self.representation = representation
        self.identifier = identifier
        super().__init__(json.dumps(
            {"attachment": "reference", "size": ref.size, "mime_type": media_type,
             **({"representation": representation} if representation != "binary" else {})},
            ensure_ascii=True, separators=(",", ":"),
        ))


    def with_origin(self, origin: ReferencedInputOrigin) -> "ReferencedContent":
        """Copy an occurrence with the actual producer without changing Asset identity."""
        if not isinstance(origin, ReferencedInputOrigin):
            raise TypeError("referenced content origin must be ReferencedInputOrigin")
        result = copy(self)
        result.origin = origin
        return result


__all__ = ["ReferencedContent", "ReferencedInputOrigin"]
