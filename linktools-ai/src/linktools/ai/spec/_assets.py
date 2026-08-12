#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Built-in Asset type bindings for declaration specifications."""

from typing import cast

from ..asset import (
    AssetRef,
    AssetTypeBinding,
    AssetVariantBinding,
    DirectoryLayout,
    SingleFileLayout,
)
from ._codec import (
    AgentSpecCodec,
    MCPServerSpecCodec,
    PromptSpecCodec,
    SkillMarkdownSpecAdapter,
    SkillMarkdownSpecCodec,
    SkillSpecCodec,
)
from ._contract import AgentSpec, MCPServerSpec, PromptSpec, SkillSpec


def builtin_asset_bindings() -> "tuple[AssetTypeBinding[object], ...]":
    """Return the immutable built-in Agent, Prompt, Skill, and MCP bindings."""
    def identity(ref: AssetRef, value: object) -> bool:
        return isinstance(value, (AgentSpec, PromptSpec, MCPServerSpec, SkillSpec)) and value.id == ref.id

    return (
        cast("AssetTypeBinding[object]", AssetTypeBinding("agent", AgentSpec, (AssetVariantBinding("json", SingleFileLayout(""), AgentSpecCodec(), "agent-spec", 1),), "json", identity, "id-v1", True)),
        cast("AssetTypeBinding[object]", AssetTypeBinding("prompt", PromptSpec, (AssetVariantBinding("json", SingleFileLayout(""), PromptSpecCodec(), "prompt-spec", 1),), "json", identity, "id-v1", True)),
        cast("AssetTypeBinding[object]", AssetTypeBinding("mcp", MCPServerSpec, (AssetVariantBinding("json", SingleFileLayout(""), MCPServerSpecCodec(), "mcp-spec", 1),), "json", identity, "id-v1", True)),
        cast("AssetTypeBinding[object]", AssetTypeBinding("skill", SkillSpec, (
            AssetVariantBinding("json", SingleFileLayout(""), SkillSpecCodec(), "skill-spec", 1),
            AssetVariantBinding("directory", DirectoryLayout("SKILL.md"), SkillMarkdownSpecCodec(), "skill-markdown", 1, SkillMarkdownSpecAdapter(), "skill-name-v1"),
        ), "directory", identity, "id-v1", True)),
    )


__all__ = ["builtin_asset_bindings"]
