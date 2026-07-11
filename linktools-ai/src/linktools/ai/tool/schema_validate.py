#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-validate tool arguments against the tool's parameter JSON schema after a
pipeline MODIFY (spec §10.2 step 6 / §12.2). A pipeline that edits arguments
must not be able to inject a payload the tool cannot safely accept: after every
MODIFY the merged arguments are checked against the same schema pydantic-ai
validated the original call against, and a mismatch fails closed."""

from typing import Any, Mapping

from ..errors import ToolDeniedError


def validate_arguments(
    arguments: "Mapping[str, Any]",
    parameters_json_schema: "Mapping[str, Any] | None",
    *,
    tool_name: str = "",
) -> None:
    """Validate ``arguments`` against ``parameters_json_schema``. No-op when no
    schema is available (the tool/adapter couldn't supply one). Raises
    ToolDeniedError on a mismatch so a bad MODIFY never reaches the handler."""
    if not parameters_json_schema:
        return
    try:
        import jsonschema
    except ImportError:  # pragma: no cover - jsonschema is a core dep here
        return
    try:
        jsonschema.validate(dict(arguments), parameters_json_schema)
    except jsonschema.ValidationError as exc:
        raise ToolDeniedError(
            f"tool {tool_name!r} arguments failed schema re-validation after "
            f"pipeline MODIFY: {exc.message}"
        ) from exc
