#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NetworkRule: denies tool calls that touch a host not on the allowlist. Pulls
the host from request.arguments["url"] (via urllib.parse.urlparse) or, failing
that, request.arguments["host"]. Calls with neither argument are unrestricted."""

from urllib.parse import urlparse

from .rule import (
    PolicyDecision,
    PolicyDecisionKind,
    ToolContext,
    ToolRequest,
)


class NetworkRule:
    def __init__(self, *, allowed_hosts: "frozenset[str]") -> None:
        self._allowed_hosts = allowed_hosts

    async def evaluate(self, request: ToolRequest, context: ToolContext) -> PolicyDecision:
        host: "str | None" = None
        url = request.arguments.get("url")
        if isinstance(url, str):
            parsed = urlparse(url)
            if parsed.hostname:
                host = parsed.hostname
        if host is None:
            raw_host = request.arguments.get("host")
            if isinstance(raw_host, str) and raw_host:
                host = raw_host
        if host is None:
            return PolicyDecision(kind=PolicyDecisionKind.ALLOW, rule_id="network-rule", reason=None)
        if host not in self._allowed_hosts:
            return PolicyDecision(
                kind=PolicyDecisionKind.DENY,
                rule_id="network-rule",
                reason=f"host {host} not allowed",
            )
        return PolicyDecision(kind=PolicyDecisionKind.ALLOW, rule_id="network-rule", reason=None)
