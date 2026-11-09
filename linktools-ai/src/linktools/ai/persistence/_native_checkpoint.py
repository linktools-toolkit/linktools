#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime checkpoint state carried by the public native snapshot format."""

from dataclasses import dataclass, fields, replace

from pydantic_ai.messages import ModelRequest
from pydantic_ai_harness.step_persistence import ContinuableSnapshot

from ..errors import AIError, ErrorCode


_STATE_KEY = "linktools.ai.checkpoint-state"
_STATE_PROTOCOL = "linktools.ai.native-checkpoint"


@dataclass(kw_only=True)
class NativeCheckpoint(ContinuableSnapshot):
    """Decoded Runtime view of one native snapshot, never an additional store."""

    output_retries_used: int

    @classmethod
    def encode(cls, snapshot: ContinuableSnapshot, *, output_retries_used: int) -> ContinuableSnapshot:
        if type(output_retries_used) is not int or output_retries_used < 0:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        messages = list(snapshot.messages) or [ModelRequest(parts=[])]
        last = messages[-1]
        # The native snapshot has no extension field. Message metadata is the
        # SDK's non-provider wire channel; wrap rather than overwrite app keys.
        messages[-1] = replace(last, metadata={_STATE_KEY: {
            "protocol": _STATE_PROTOCOL,
            "version": 1,
            "empty_messages": not snapshot.messages,
            "run_id": snapshot.run_id,
            "step_index": snapshot.step_index,
            "state": snapshot.state,
            "timestamp": snapshot.timestamp.isoformat(),
            "idempotency_key": snapshot.idempotency_key,
            "output_retries_used": output_retries_used,
            "metadata": last.metadata,
        }})
        return ContinuableSnapshot(**{
            field.name: messages if field.name == "messages" else getattr(snapshot, field.name)
            for field in fields(ContinuableSnapshot)
        })

    @classmethod
    def decode(cls, snapshot: ContinuableSnapshot) -> ContinuableSnapshot:
        # Decoded metadata belongs to the application, even when it uses the
        # same protocol key. Reader adapters may compose without decoding twice.
        if isinstance(snapshot, cls):
            return snapshot
        messages = list(snapshot.messages)
        outer = None if not messages else messages[-1].metadata
        if outer is not None and not isinstance(outer, dict):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if outer is None or _STATE_KEY not in outer:
            # An upstream-authored snapshot has no Runtime retry fact. It can
            # supply history, but cannot restore the same execution budget.
            return snapshot
        state = outer[_STATE_KEY]
        if not isinstance(state, dict) or state.get("protocol") != _STATE_PROTOCOL:
            return snapshot
        last = messages[-1]
        if set(outer) != {_STATE_KEY}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (set(state) != {
            "protocol", "version", "empty_messages", "run_id", "step_index", "state", "timestamp", "idempotency_key",
            "output_retries_used", "metadata",
        } or type(state["version"]) is not int or state["version"] != 1
                or type(state["empty_messages"]) is not bool
                or state["run_id"] != snapshot.run_id
                or type(state["step_index"]) is not int or state["step_index"] != snapshot.step_index
                or state["timestamp"] != snapshot.timestamp.isoformat()
                or state["state"] != snapshot.state or state["idempotency_key"] != snapshot.idempotency_key
                or type(state["output_retries_used"]) is not int or state["output_retries_used"] < 0
                or state["metadata"] is not None and not isinstance(state["metadata"], dict)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if state["empty_messages"]:
            if (len(messages) != 1 or not isinstance(last, ModelRequest)
                    or last.parts or state["metadata"] is not None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            messages = []
        else:
            messages[-1] = replace(last, metadata=state["metadata"])
        return cls(**{
            field.name: messages if field.name == "messages" else getattr(snapshot, field.name)
            for field in fields(ContinuableSnapshot)
        }, output_retries_used=state["output_retries_used"])


__all__ = ["NativeCheckpoint"]
