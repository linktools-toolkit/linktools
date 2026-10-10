#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified MCP authoring, connection fields, and durable wire contracts."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from linktools.ai.asset import AssetKey
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.spec import MCPServerSpec, MCPServerSpecAdapter, MCPServerSpecCodec
from linktools.ai.workspace import LocalSandbox, ReadOnlySandboxPolicy


def _decode_author(payload: dict[str, object], entry: str) -> MCPServerSpec:
    adapter = MCPServerSpecAdapter()
    if entry == "json":
        return adapter.decode_json(json.dumps(payload).encode(), package_id="server")
    if entry == "yaml":
        return adapter.decode_yaml(yaml.safe_dump(payload).encode(), package_id="server")
    return adapter.decode_config(
        json.dumps({"mcpServers": {"server": payload}}).encode()
    )[0]


@pytest.mark.parametrize("entry", ("json", "yaml", "shared"))
@pytest.mark.parametrize(
    ("selector", "transport"),
    (
        ({}, "streamable-http"),
        ({"type": "http"}, "streamable-http"),
        ({"transport": "http"}, "streamable-http"),
        ({"transport": "streamable-http"}, "streamable-http"),
        ({"type": "sse"}, "sse"),
    ),
)
def test_remote_authoring_entries_resolve_identical_connection_fields(
    entry: str,
    selector: dict[str, object],
    transport: str,
) -> None:
    payload = {
        "id": "server",
        "revision": 7,
        "url": "https://example.test/events",
        "headers": {"Authorization": "Bearer token"},
        "init_timeout": 1.5,
        "read_timeout": 4,
        "future_note": {"enabled": True},
        **selector,
    }
    expected = MCPServerSpec(
        "server",
        transport=transport,
        url="https://example.test/events",
        headers={"Authorization": "Bearer token"},
        init_timeout=1.5,
        read_timeout=4,
        revision=7,
    )
    actual = _decode_author(payload, entry)
    codec = MCPServerSpecCodec()
    assert codec.to_wire_payload(actual) == codec.to_wire_payload(expected)
    assert actual.resource is None
    assert codec.to_binding_payload(actual, None) == codec.to_binding_payload(expected, None)


@pytest.mark.parametrize("entry", ("json", "yaml", "shared"))
@pytest.mark.parametrize(
    "invalid",
    (
        {"resource": None},
        {"command": None},
        {"args": []},
        {"env": {}},
        {"transport": "sse", "type": "sse"},
        {"transport": "unknown"},
        {"transport": []},
        {"transport": None},
        {"headers": {"Authorization": 42}},
        {"revision": True},
        {"resource_versions": []},
        {"asset_source_id": "assets"},
        {"execution_policy": {"version": 1, "boundary": "host-network"}},
    ),
)
def test_authoring_rejects_known_invalid_and_runtime_owned_fields(
    entry: str, invalid: dict[str, object]
) -> None:
    with pytest.raises(AIError) as error:
        _decode_author({"url": "https://example.test/mcp", **invalid}, entry)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.parametrize("entry", ("json", "yaml", "shared"))
@pytest.mark.parametrize("remote_field", ({"url": None}, {"headers": {}}))
def test_stdio_authoring_rejects_explicit_remote_fields(
    entry: str, remote_field: dict[str, object]
) -> None:
    with pytest.raises(AIError) as error:
        _decode_author({"command": "python", **remote_field}, entry)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.parametrize("entry", ("json", "yaml", "shared"))
def test_authoring_requires_matching_explicit_server_identity(entry: str) -> None:
    with pytest.raises(AIError) as error:
        _decode_author({"id": "another", "command": "python"}, entry)
    assert error.value.code is ErrorCode.ASSET_CONTENT_MISMATCH


def test_stdio_resource_defaults_preserve_explicit_values() -> None:
    adapter = MCPServerSpecAdapter()
    implicit = adapter.decode_json(b'{"command":"python"}', package_id="server")
    assert implicit.resource == AssetKey("mcp", "server")
    for resource in (None, {"kind": "mcp", "id": "shared/resources"}):
        for entry in ("json", "yaml", "shared"):
            server = _decode_author({"command": "python", "resource": resource}, entry)
            assert server.resource == (
                None if resource is None else AssetKey("mcp", "shared/resources")
            )
    nonpackage = adapter.decode_json(b'{"id":"server","command":"python"}')
    assert nonpackage.resource is None


def test_shared_revision_defaults_do_not_override_explicit_revision() -> None:
    servers = MCPServerSpecAdapter().decode_config(
        b'{"mcpServers":{"explicit":{"command":"python","revision":8},'
        b'"default":{"command":"python"}}}',
        revision=3,
    )
    assert {server.id: server.revision for server in servers} == {"explicit": 8, "default": 3}


@pytest.mark.parametrize("field", ("init_timeout", "read_timeout"))
@pytest.mark.parametrize("value", (True, 0, -0.5, float("nan"), float("inf"), float("-inf"), "1"))
def test_mcp_connection_timeouts_require_finite_positive_numbers(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError)):
        MCPServerSpec("server", "python", **{field: value})
    payload = {
        "version": 1,
        "id": "server",
        "transport": "stdio",
        "command": "python",
        field: value,
    }
    with pytest.raises(AIError) as error:
        MCPServerSpecCodec().from_payload(payload)
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    with pytest.raises(AIError) as error:
        _decode_author(payload, "json")
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.parametrize("transport", ("stdio", "streamable-http", "sse"))
def test_timeouts_round_trip_but_do_not_change_semantic_contract(transport: str) -> None:
    server = (
        MCPServerSpec("server", "python")
        if transport == "stdio"
        else MCPServerSpec("server", transport=transport, url="https://example.test/mcp")
    )
    codec = MCPServerSpecCodec()
    tuned = replace(server, init_timeout=0.25, read_timeout=10)
    assert tuned == server
    assert codec.to_contract_payload(tuned) == codec.to_contract_payload(server)
    assert codec.to_binding_payload(tuned, None) == codec.to_binding_payload(server, None)
    restored = codec.decode(codec.encode(tuned))
    assert restored.init_timeout == 0.25
    assert restored.read_timeout == 10
    default = codec.decode(codec.encode(server))
    assert default.init_timeout is None
    assert default.read_timeout is None
    explicit_null = codec.from_payload(
        {**codec.to_wire_payload(server), "init_timeout": None, "read_timeout": None}
    )
    assert explicit_null.init_timeout is None
    assert explicit_null.read_timeout is None
    for field in ("init_timeout", "read_timeout"):
        with pytest.raises(AIError) as error:
            codec.decode_binding_payload(
                {**codec.to_binding_payload(server, None), field: 1}, declaration=server
            )
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("field", ({"transport": "http"}, {"type": "sse"}, {"transport": []}))
def test_wire_requires_canonical_transport_fields(field: dict[str, object]) -> None:
    with pytest.raises(AIError) as error:
        MCPServerSpecCodec().from_payload(
            {
                "version": 1,
                "id": "server",
                "transport": "streamable-http",
                "url": "https://example.test/mcp",
                **field,
            }
        )
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.parametrize("entry", ("json", "yaml", "shared"))
def test_environment_expansion_is_single_pass_and_connection_values_only(
    monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    monkeypatch.setenv("MCP_HOST", "example.test")
    monkeypatch.setenv("MCP_TOKEN", "${UNCHANGED}")
    remote = _decode_author(
        {
            "url": "https://${MCP_HOST}/events",
            "headers": {"${HEADER_NAME}": "Bearer ${MCP_TOKEN} $$ $NAME $${LITERAL}"},
        },
        entry,
    )
    assert remote.url == "https://example.test/events"
    assert dict(remote.headers) == {"${HEADER_NAME}": "Bearer ${UNCHANGED} $ $NAME ${LITERAL}"}
    local = _decode_author(
        {
            "command": "${COMMAND}",
            "args": ["${ARGUMENT}"],
            "resource": {"kind": "mcp", "id": "${RESOURCE}"},
            "env": {"${ENV_NAME}": "${MCP_TOKEN} $$"},
        },
        entry,
    )
    assert local.command == "${COMMAND}"
    assert local.args == ("${ARGUMENT}",)
    assert local.resource == AssetKey("mcp", "${RESOURCE}")
    assert dict(local.env) == {"${ENV_NAME}": "${UNCHANGED} $"}
    monkeypatch.setenv("MCP_TOKEN", "rotated")
    assert remote.headers["${HEADER_NAME}"] == "Bearer ${UNCHANGED} $ $NAME ${LITERAL}"
    assert local.env["${ENV_NAME}"] == "${UNCHANGED} $"


@pytest.mark.parametrize(
    "value",
    ("prefix-secret ${MCP_MISSING}", "prefix-secret ${MCP_MISSING", "${MCP_MISSING:-secret}", "${}"),
)
@pytest.mark.parametrize("field", ("url", "headers", "env"))
def test_environment_reference_errors_are_typed_and_exclude_values(
    monkeypatch: pytest.MonkeyPatch, value: str, field: str
) -> None:
    monkeypatch.delenv("MCP_MISSING", raising=False)
    payload = (
        {"command": "python", "env": {"TOKEN": value}}
        if field == "env"
        else {"url": value}
        if field == "url"
        else {"url": "https://example.test/mcp", "headers": {"TOKEN": value}}
    )
    with pytest.raises(AIError) as error:
        _decode_author(payload, "json")
    assert error.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID
    details = error.value.safe_details
    assert details["server_id"] == "server"
    assert details["field"] == ("url" if field == "url" else f"{field}.TOKEN")
    assert set(details) <= {"server_id", "field", "variable"}
    assert "secret" not in str(error.value)
    assert "secret" not in json.dumps(details)
    assert error.value.__cause__ is None


def test_python_and_wire_never_expand_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TOKEN", "must-not-be-read")
    server = MCPServerSpec(
        "${MCP_ID}",
        transport="streamable-http",
        url="https://example.test/${MCP_PATH}",
        headers={"Authorization": "${MCP_TOKEN} $$"},
    )
    codec = MCPServerSpecCodec()
    restored = codec.decode(codec.encode(server))
    assert restored.id == "${MCP_ID}"
    assert restored.url == "https://example.test/${MCP_PATH}"
    assert restored.headers["Authorization"] == "${MCP_TOKEN} $$"
    declaration = MCPServerSpecAdapter().decode_json(
        b'{"id":"${MCP_ID}","command":"python"}'
    )
    assert declaration.id == "${MCP_ID}"


@pytest.mark.parametrize("read_policy_stdio", (False, True))
def test_resource_working_directory_is_durable_without_physical_paths(
    read_policy_stdio: bool,
) -> None:
    from linktools.ai.runtime._mcp import _mcp_execution_policy

    server = MCPServerSpec(
        "server", "python", ("server.py",), AssetKey("mcp", "server"),
    )
    codec = MCPServerSpecCodec()
    sandbox = LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)),
        allow_host_stdio_with_read_policy=True,
    ) if read_policy_stdio else None
    payload = codec.to_binding_payload(
        server, (), asset_source_id="assets",
        execution_policy=_mcp_execution_policy(server, sandbox),
    )
    assert payload["execution_policy"] == {
        "version": 1, "boundary": "host-stdio", "cwd": "resource",
    }
    assert codec.decode_binding_payload(payload, declaration=server) == ()
    assert payload["args"] == ["server.py"]


@pytest.mark.parametrize("policy", (
    {"version": True, "boundary": "host-stdio"},
    {"version": 1.0, "boundary": "host-network"},
    {"version": 1, "boundary": [], "cwd": "resource"},
    {"version": 1, "boundary": "host-stdio", "cwd": None},
    {"version": 1, "boundary": "host-stdio", "cwd": "/tmp/private"},
    {"version": 1, "boundary": "host-network", "cwd": "resource"},
))
def test_execution_working_directory_policy_rejects_invalid_semantics(
    policy: dict[str, object],
) -> None:
    with pytest.raises(AIError) as raised:
        MCPServerSpecCodec().to_binding_payload(
            MCPServerSpec("server", "python"), None, execution_policy=policy,
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("read_policy_stdio", (False, True))
async def test_old_workspace_cwd_binding_cannot_silently_change_execution(
    tmp_path: Path,
    read_policy_stdio: bool,
) -> None:
    from linktools.ai.runtime._mcp import (
        _MCPBinding, _MCPProjection, materialize_mcp_capabilities,
    )
    from linktools.ai.spec import mcp_server_selector

    server = MCPServerSpec(
        "server", "python", ("server.py",), AssetKey("mcp", "server"),
    )
    sandbox = LocalSandbox(
        read_policy=ReadOnlySandboxPolicy(("visible.txt",)),
        allow_host_stdio_with_read_policy=True,
    ) if read_policy_stdio else None
    with pytest.raises(AIError) as raised:
        await materialize_mcp_capabilities(
            (server,), (mcp_server_selector(server.id),),
            sandbox=sandbox, sandbox_session=None, host_cwd=str(tmp_path),
            bindings={server.id: _MCPBinding(
                (), "assets", {"version": 1, "boundary": "host-stdio"},
            )},
            projections={server.id: _MCPProjection(server.id, server.args, (), str(tmp_path))},
            tool_operations=None, tool_metrics=None,
        )
    assert raised.value.code is ErrorCode.CAPABILITY_POLICY_CONFLICT
