#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Typed evidence and the shared Task/Agent scoring wire contract."""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from ..asset import AssetVersionRef
from ..core import ImmutableJsonMapping, JsonValue, UsageMetrics, canonical_sha256, validate_idempotency_key
from ..storage import ObjectRef
from ._contracts import InlineValue, ScorerContract, TargetTrialRef


@dataclass(frozen=True, slots=True)
class ExecutionSubjectRef:
    namespace: str
    tenant_id: str
    execution_id: str

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"kind": "execution", "namespace": self.namespace, "tenant_id": self.tenant_id,
                "execution_id": self.execution_id}


@dataclass(frozen=True, slots=True)
class GraphSubjectRef:
    namespace: str
    tenant_id: str
    graph_id: str

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"kind": "graph", "namespace": self.namespace, "tenant_id": self.tenant_id,
                "graph_id": self.graph_id}


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    namespace: str
    tenant_id: str
    evidence_id: str
    digest: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in (self.namespace, self.tenant_id, self.evidence_id)):
            raise ValueError("evidence identity is incomplete")
        if not isinstance(self.digest, str) or re.fullmatch(r"[0-9a-f]{64}", self.digest) is None:
            raise ValueError("evidence digest is invalid")

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"namespace": self.namespace, "tenant_id": self.tenant_id,
                "evidence_id": self.evidence_id, "digest": self.digest}


@dataclass(frozen=True, slots=True)
class ExecutionTargetEvidence:
    subject: ExecutionSubjectRef
    status: str
    output: InlineValue | None = None
    error_code: str | None = None

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"kind": "execution", "subject": self.subject.to_mapping(), "status": str(self.status),
                "output": None if self.output is None else self.output.to_mapping(), "error_code": self.error_code}


@dataclass(frozen=True, slots=True)
class GraphOutputItem:
    status: str
    value: InlineValue | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.status != "succeeded" and not self.reason:
            raise ValueError("unsuccessful graph output requires a reason")
        if self.status == "succeeded" and self.value is None:
            raise ValueError("successful graph output requires a value, including explicit JSON null")

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"status": str(self.status), "value": None if self.value is None else self.value.value,
                "reason": self.reason}


@dataclass(frozen=True, slots=True)
class GraphTargetEvidence:
    subject: GraphSubjectRef
    status: str
    outputs: Mapping[str, GraphOutputItem] = field(default_factory=dict)
    node_statuses: Mapping[str, str] = field(default_factory=dict)
    error_code: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(self, "node_statuses", MappingProxyType(dict(self.node_statuses)))

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"kind": "graph", "subject": self.subject.to_mapping(), "status": str(self.status),
                "outputs": {name: item.to_mapping() for name, item in self.outputs.items()},
                "node_statuses": {name: str(status) for name, status in self.node_statuses.items()},
                "error_code": self.error_code}


def _usage_mapping(value: UsageMetrics) -> dict[str, JsonValue]:
    return {"model_requests": value.model_requests, "tool_calls": value.tool_calls,
            "input_tokens": value.input_tokens, "output_tokens": value.output_tokens,
            "cache_read_tokens": value.cache_read_tokens, "cache_write_tokens": value.cache_write_tokens}


@dataclass(frozen=True, slots=True)
class ModelUsage:
    provider: str
    model: str
    usage: UsageMetrics
    complete: bool

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"provider": self.provider, "model": self.model, "usage": _usage_mapping(self.usage),
                "complete": self.complete}


@dataclass(frozen=True, slots=True)
class EvidenceAttachmentRef:
    attachment_id: str
    media_type: str
    object_ref: ObjectRef

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (self.attachment_id, self.media_type)):
            raise ValueError("evidence attachment requires an identity and media type")

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"kind": "runtime_content", "attachment_id": self.attachment_id, "media_type": self.media_type,
                "digest": self.object_ref.digest, "size": self.object_ref.size}


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    ref: EvidenceRef
    trial: TargetTrialRef
    target: ExecutionTargetEvidence | GraphTargetEvidence
    input: InlineValue | None = None
    trace: tuple[Mapping[str, JsonValue], ...] = ()
    attachments: tuple[AssetVersionRef | EvidenceAttachmentRef, ...] = ()
    usage: UsageMetrics | None = None
    usage_complete: bool = False
    cutoff: Mapping[str, JsonValue] = field(default_factory=dict)
    source_ref: EvidenceRef | None = None
    model_usage: tuple[ModelUsage, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "trace", tuple(ImmutableJsonMapping(item) for item in self.trace))
        object.__setattr__(self, "attachments", tuple(self.attachments))
        attachment_ids = [item.attachment_id for item in self.attachments if isinstance(item, EvidenceAttachmentRef)]
        if len(set(attachment_ids)) != len(attachment_ids):
            raise ValueError("duplicate evidence attachment identity")
        object.__setattr__(self, "cutoff", ImmutableJsonMapping(self.cutoff))
        object.__setattr__(self, "model_usage", tuple(self.model_usage))

    def content_mapping(self) -> dict[str, JsonValue]:
        usage = self.usage
        return {"trial": self.trial.to_mapping(), "target": self.target.to_mapping(),
                "input": None if self.input is None else self.input.to_mapping(),
                "trace": [dict(item) for item in self.trace],
                "attachments": [item.to_mapping() if isinstance(item, EvidenceAttachmentRef)
                                else {"kind": "asset", "ref": item.to_payload()} for item in self.attachments],
                "usage": None if usage is None else _usage_mapping(usage),
                "usage_complete": self.usage_complete, "cutoff": dict(self.cutoff),
                "source_ref": None if self.source_ref is None else self.source_ref.to_mapping(),
                "model_usage": [item.to_mapping() for item in self.model_usage]}

    @property
    def digest(self) -> str:
        return canonical_sha256(self.content_mapping())


class ScoreNotApplicable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["not_applicable"] = "not_applicable"
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def meaningful_reason(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("NA requires a reason")
        return value


class ScoreBundle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    dimensions: Mapping[str, float | ScoreNotApplicable]
    rationale: str | None = None
    evidence_ids: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    @field_validator("dimensions", mode="before")
    @classmethod
    def validate_numbers(cls, value: object) -> object:
        if not isinstance(value, Mapping) or not value:
            raise ValueError("score requires dimensions")
        for name, item in value.items():
            if not isinstance(name, str) or not name:
                raise ValueError("score dimension requires a name")
            if isinstance(item, bool) or isinstance(item, (int, float)) and not math.isfinite(item):
                raise ValueError("score must be a finite number or an explicit NA result")
            if isinstance(item, str):
                raise ValueError("string scores are not numeric scores")
        return value

    @field_validator("dimensions")
    @classmethod
    def freeze_dimensions(cls, value: Mapping[str, float | ScoreNotApplicable]) -> Mapping[str, float | ScoreNotApplicable]:
        return MappingProxyType(dict(value))

    @field_serializer("dimensions")
    def serialize_dimensions(self, value: Mapping[str, float | ScoreNotApplicable]) -> dict[str, object]:
        return {name: item.model_dump(mode="json") if isinstance(item, ScoreNotApplicable) else item
                for name, item in value.items()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, JsonValue]) -> "ScoreBundle":
        return cls.model_validate(dict(value))

    def to_mapping(self) -> dict[str, JsonValue]:
        return cast("dict[str, JsonValue]", self.model_dump(mode="json"))

    def validate_contract(self, contract: ScorerContract, *, evidence_ids: tuple[str, ...] = ()) -> None:
        if set(self.dimensions) != {item.name for item in contract.dimensions}:
            raise ValueError("score dimensions do not match scorer contract")
        for dimension in contract.dimensions:
            value = self.dimensions[dimension.name]
            if isinstance(value, ScoreNotApplicable):
                continue
            if dimension.minimum is not None and value < dimension.minimum or dimension.maximum is not None and value > dimension.maximum:
                raise ValueError("score is outside the declared dimension range")
        if not set(self.evidence_ids).issubset(evidence_ids):
            raise ValueError("score cites evidence outside the authorized bundle")


@dataclass(frozen=True, slots=True)
class HumanScoreRequest:
    trial_id: str
    scorer_slot_id: str
    evidence_ref: EvidenceRef
    score: ScoreBundle
    idempotency_key: str

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value.strip() for value in (
            self.trial_id, self.scorer_slot_id,
        )):
            raise ValueError("human score request identity is incomplete")
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True, init=False)
class ScoringInput:
    trial: TargetTrialRef
    target: ExecutionTargetEvidence | GraphTargetEvidence
    content: Mapping[str, JsonValue]
    evidence_ref: EvidenceRef
    candidate_label: str

    def __init__(
        self, trial: TargetTrialRef, target: ExecutionTargetEvidence | GraphTargetEvidence,
        expected: JsonValue, expected_present: bool, rubric: JsonValue,
        config: Mapping[str, JsonValue], evidence_ref: EvidenceRef, candidate_label: str,
        *, rubric_present: bool = False, target_input: JsonValue = None,
    ) -> None:
        if not isinstance(expected_present, bool) or not isinstance(rubric_present, bool):
            raise ValueError("input presence must be boolean")
        object.__setattr__(self, "trial", trial)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "evidence_ref", evidence_ref)
        object.__setattr__(self, "candidate_label", candidate_label)
        object.__setattr__(self, "content", ImmutableJsonMapping({
            "expected": expected, "expected_present": expected_present,
            "rubric": rubric, "rubric_present": rubric_present, "config": dict(config),
            "target_input": target_input,
        }))

    @property
    def target_input(self) -> JsonValue:
        return self.content["target_input"]

    @property
    def expected(self) -> JsonValue:
        return self.content["expected"]

    @property
    def expected_present(self) -> bool:
        return cast(bool, self.content["expected_present"])

    @property
    def rubric(self) -> JsonValue:
        return self.content["rubric"]

    @property
    def rubric_present(self) -> bool:
        return cast(bool, self.content["rubric_present"])

    @property
    def config(self) -> Mapping[str, JsonValue]:
        return _mapping(self.content["config"])

    @property
    def evidence_digest(self) -> str:
        return self.evidence_ref.digest

    @property
    def target_status(self) -> str:
        return self.target.status

    @property
    def target_output(self) -> JsonValue:
        if isinstance(self.target, ExecutionTargetEvidence):
            return None if self.target.output is None else self.target.output.value
        return {name: item.to_mapping() for name, item in self.target.outputs.items()}

    def to_mapping(self) -> dict[str, JsonValue]:
        return {"kind": "evaluation-scoring-input", "trial": self.trial.to_mapping(),
                "target": self.target.to_mapping(), **dict(self.content),
                "evidence_ref": self.evidence_ref.to_mapping(), "candidate_label": self.candidate_label}

    @classmethod
    def from_mapping(cls, value: Mapping[str, JsonValue]) -> "ScoringInput":
        required = {"kind", "trial", "target", "expected", "expected_present", "rubric", "rubric_present",
                    "config", "evidence_ref", "candidate_label", "target_input"}
        if set(value) != required or value["kind"] != "evaluation-scoring-input":
            raise ValueError("invalid scoring input contract")
        trial = _mapping(value["trial"])
        evidence = _mapping(value["evidence_ref"])
        for field_name in ("expected_present", "rubric_present"):
            if not isinstance(value[field_name], bool):
                raise ValueError("input presence must be boolean")
        return cls(TargetTrialRef(_text(trial, "target_experiment_id"), _text(trial, "trial_id")),
                   _target_from_mapping(_mapping(value["target"])),
                   value["expected"], cast(bool, value["expected_present"]), value["rubric"],
                   _mapping(value["config"]),
                   EvidenceRef(_text(evidence, "namespace"), _text(evidence, "tenant_id"),
                               _text(evidence, "evidence_id"), _text(evidence, "digest")),
                   _text(value, "candidate_label"), rubric_present=cast(bool, value["rubric_present"]),
                   target_input=value["target_input"])


def _mapping(value: JsonValue) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a JSON object")
    return value


def _text(value: Mapping[str, JsonValue], name: str) -> str:
    item = value.get(name)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{name} must be non-empty text")
    return item


def _optional_text(value: Mapping[str, JsonValue], name: str) -> str | None:
    item = value.get(name)
    if item is not None and not isinstance(item, str):
        raise ValueError(f"{name} must be text or null")
    return item


def _target_from_mapping(value: Mapping[str, JsonValue]) -> ExecutionTargetEvidence | GraphTargetEvidence:
    subject = _mapping(value["subject"])
    namespace, tenant = _text(subject, "namespace"), _text(subject, "tenant_id")
    if value.get("kind") == "execution":
        if set(value) != {"kind", "subject", "status", "output", "error_code"} or subject.get("kind") != "execution":
            raise ValueError("invalid execution evidence")
        output = value["output"]
        if output is not None:
            payload = _mapping(output)
            if set(payload) != {"kind", "value"} or payload["kind"] != "json":
                raise ValueError("invalid execution result value")
            wrapped = InlineValue.from_value(payload["value"])
        else:
            wrapped = None
        return ExecutionTargetEvidence(ExecutionSubjectRef(namespace, tenant, _text(subject, "execution_id")),
                                       _text(value, "status"), wrapped, _optional_text(value, "error_code"))
    if value.get("kind") != "graph" or set(value) != {"kind", "subject", "status", "outputs", "node_statuses", "error_code"} or subject.get("kind") != "graph":
        raise ValueError("unknown target evidence kind")
    outputs: dict[str, GraphOutputItem] = {}
    for name, item in _mapping(value["outputs"]).items():
        payload = _mapping(item)
        if set(payload) != {"status", "value", "reason"}:
            raise ValueError("invalid graph output")
        status = _text(payload, "status")
        outputs[name] = GraphOutputItem(status, InlineValue.from_value(payload["value"]) if status == "succeeded" else None,
                                        _optional_text(payload, "reason"))
    statuses = _mapping(value["node_statuses"])
    return GraphTargetEvidence(GraphSubjectRef(namespace, tenant, _text(subject, "graph_id")),
                               _text(value, "status"), outputs,
                               {name: _text(statuses, name) for name in statuses}, _optional_text(value, "error_code"))


__all__ = ["EvidenceAttachmentRef", "EvidenceBundle", "EvidenceRef", "ExecutionSubjectRef", "ExecutionTargetEvidence", "GraphOutputItem", "HumanScoreRequest", "ModelUsage",
           "GraphSubjectRef", "GraphTargetEvidence", "ScoreBundle", "ScoreNotApplicable", "ScoringInput"]
