#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Built-in Asset type bindings for declaration specifications."""

from dataclasses import replace
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
    SkillMarkdownSpecAdapter,
    SkillMarkdownSpecCodec,
    SkillSpecCodec,
    retarget_skill_markdown,
)
from ._contract import AgentSpec, MCPServerSpec, SkillSpec


def builtin_asset_bindings() -> "tuple[AssetTypeBinding[object], ...]":
    """Return the immutable built-in Agent, Skill, and MCP bindings."""

    def identity(ref: AssetRef, value: object) -> bool:
        return isinstance(value, (AgentSpec, MCPServerSpec, SkillSpec)) and value.id == ref.id

    def retarget(source: AssetRef, target: AssetRef, variant: str, value: object) -> object:
        del source
        if isinstance(value, AgentSpec):
            return replace(value, id=target.id)
        if isinstance(value, MCPServerSpec):
            return replace(value, id=target.id)
        if isinstance(value, SkillSpec):
            content = value.content
            if variant == "directory":
                content = retarget_skill_markdown(content, target.id.rsplit("/", 1)[-1])
            return replace(value, id=target.id, content=content)
        raise TypeError("asset retarget value type is invalid")

    return (
        cast(
            "AssetTypeBinding[object]",
            AssetTypeBinding(
                "agent",
                AgentSpec,
                (AssetVariantBinding("json", SingleFileLayout(""), AgentSpecCodec(), "agent-spec", 1),),
                "json",
                identity,
                "id-v1",
                True,
                retarget,
                "id-retarget-v1",
            ),
        ),
        cast(
            "AssetTypeBinding[object]",
            AssetTypeBinding(
                "mcp",
                MCPServerSpec,
                (AssetVariantBinding("json", SingleFileLayout(""), MCPServerSpecCodec(), "mcp-spec", 1),),
                "json",
                identity,
                "id-v1",
                True,
                retarget,
                "id-retarget-v1",
            ),
        ),
        cast(
            "AssetTypeBinding[object]",
            AssetTypeBinding(
                "skill",
                SkillSpec,
                (
                    AssetVariantBinding("json", SingleFileLayout(""), SkillSpecCodec(), "skill-spec", 1),
                    AssetVariantBinding(
                        "directory",
                        DirectoryLayout("SKILL.md"),
                        SkillMarkdownSpecCodec(),
                        "skill-markdown",
                        1,
                        SkillMarkdownSpecAdapter(),
                        "skill-name-v1",
                    ),
                ),
                "directory",
                identity,
                "id-v1",
                True,
                retarget,
                "id-retarget-v1",
            ),
        ),
    )


__all__ = ["builtin_asset_bindings"]
