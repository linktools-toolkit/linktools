#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared v1 execution-owner fields for persistence fixtures."""

from linktools.ai.runtime.state._contracts import StoredUserInput
from linktools.ai.storage import StoredPayload


def execution_owner_fields(prompt: str = "prompt") -> dict[str, object]:
    return {
        "principal_id": "principal",
        "principal_kind": "service",
        "stored_user_input": StoredUserInput(
            "text",
            StoredPayload.inline_text(prompt),
        ),
    }
