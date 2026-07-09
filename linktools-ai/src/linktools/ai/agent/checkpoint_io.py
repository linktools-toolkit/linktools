#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Serialization helpers for RunCheckpoint.payload. Delegates to pydantic-ai's
ModelMessagesTypeAdapter (messages.py in pydantic_ai 1.107.0) so a future
pydantic-ai format upgrade is transparent. The single (de)serialization seam."""

from typing import Sequence

from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter


def serialize_messages(messages: "Sequence[ModelMessage]") -> bytes:
    return ModelMessagesTypeAdapter.dump_json(list(messages))


def deserialize_messages(data: bytes) -> "list[ModelMessage]":
    return ModelMessagesTypeAdapter.validate_json(data)
