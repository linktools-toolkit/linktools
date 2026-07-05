#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Model config resolution for the `lt ai chat` command."""

import os

from linktools.cli import CommandError
from linktools.ai.model.registry import RuntimeModelConfig


def resolve_model_config(
    model: "str | None",
    base_url: "str | None",
    api_key: "str | None",
) -> RuntimeModelConfig:
    """Build a `RuntimeModelConfig` from CLI flags, falling back to env vars.

    Flags win over environment variables. Raises `CommandError` if `base_url`
    or `api_key` are unset after both sources are checked.
    """
    resolved_model = model or os.environ.get("OPENAI_MODEL") or ""
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")

    if not resolved_base_url:
        raise CommandError(
            "no base url provided: pass --base-url or set OPENAI_BASE_URL"
        )
    if not resolved_api_key:
        raise CommandError(
            "no api key provided: pass --api-key or set OPENAI_API_KEY"
        )

    return RuntimeModelConfig(
        model_type="standard",
        protocol="openai",
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        auth_token=None,
        timeout_seconds=300,
        raw={},
    )


def validate_session_id(session_id: str) -> str:
    """Reject session ids that could escape the sessions directory when joined
    into a filesystem path. Raises CommandError with a clear message; returns
    session_id unchanged if it's safe.
    """
    is_unsafe = (
        not session_id
        or "/" in session_id
        or "\\" in session_id
        or session_id in (".", "..")
    )
    if is_unsafe:
        raise CommandError(
            f'invalid --session value "{session_id}": must not contain path separators or ".."'
        )
    return session_id
