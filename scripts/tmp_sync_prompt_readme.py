#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Temporary README synchronizer for the prompt partition implementation."""

from pathlib import Path


path = Path("linktools-ai/README.md")
text = path.read_text()
old = """Model-visible control text is composed as a stable front-of-request instruction prefix plus the current repository contribution. `AgentSpec.system_prompt` remains the standing Agent prompt; literal Agent instructions, selected Skill/Subagent guidance, explicit preloaded Skill content, Workspace usage guidance, and other static capability instructions remain stable for the run. Repository instructions from the applicable `AGENTS.md` and `.linktools/rules` sources are refreshed only when the existing path-scoped repository boundary activates additional rules. Provider adapters may serialize these instruction parts as `system`, `developer`, or another supported instruction channel; the Python field used to declare a contribution does not by itself define its provider role.

Loaded Skill bodies, Memory content, current plans, user input, tool results, and retry/error feedback keep their existing contextual positions instead of becoming permanent system guidance. Workspace instruction text describes only stable usage constraints; authorization and approval remain Runtime-enforced behavior and are not inferred from prompt text. Model-interaction history can therefore record the same stable instruction content on multiple requests without that meaning LinkTools appended a new historical instruction each time.
"""
new = """Model-visible control text is partitioned by lifecycle. Binding-static Agent instructions, Skill catalog/preloads, and Workspace guidance form the most stable prefix. Execution-static capability guidance and the repository instructions already active when the Execution starts follow that prefix and remain fixed for that Execution. Repository sources first discovered later through declared workspace path fields are kept in one refreshable overlay; only that overlay changes when a new scope becomes effective. Provider adapters may serialize these instruction parts as `system`, `developer`, or another supported instruction channel, and the physical request may still carry the same fixed prefix on every model call so provider prompt caching can reuse it.

The fixed prefix is not appended as a new conversation message on each model call. Lazy Skill bodies/resources, Memory contents, the current-plan reminder, user input, tool and Subagent results, attachments, and retry/error feedback remain dynamic request or conversation context. Memory guidance is fixed only for an Execution that enables Memory; Memory contents stay pull-based through the Memory tools, so updates from another Execution are observed when the Agent reads/searches Memory again rather than being pushed into the standing prompt. Workspace instruction text describes stable usage constraints only; authorization and approval remain Runtime-enforced behavior. Request snapshots may therefore contain the same fixed instruction parts repeatedly without representing repeated historical messages.
"""
if text.count(old) != 1:
    raise SystemExit("README prompt lifecycle paragraph changed unexpectedly")
path.write_text(text.replace(old, new, 1))
