#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""``lt ai init`` business logic.

Scaffolds a project's ``.linktools/``: the default config + a default/reviewer
agent + a code-review skill and the skill-creator example (with its private
grader/comparator/analyzer agents) + a disabled GitHub MCP template. Existing
files are never overwritten so re-running ``init`` on a customized project is
safe."""

from pathlib import Path

from linktools.core import environ

_CONFIG_YAML = """\
version: 1

default_agent: default
default_session: main

subagents:
  max_depth: 3
  max_concurrency: 4
  default_timeout_seconds: 120

skills:
  allow_private_agents: true

mcp:
  allow_wildcard: false
  discovery_mode: strict
"""

_DEFAULT_AGENT_MD = """\
---
name: default

model:
  primary: standard
  request_retries: 1
  timeout_seconds: 120

tools:
  - kind: builtin
    name: file-read
  - kind: builtin
    name: file-write
  - kind: builtin
    name: terminal
  - kind: skill
    name: "*"
  - kind: subagent
    name: reviewer
---

Inspect the project before modifying files.
Use skills when they match the task.
Ask for approval before high-risk actions.
"""

_REVIEWER_AGENT_MD = """\
---
name: reviewer

model:
  primary: standard
  request_retries: 1
  timeout_seconds: 120

tools:
  - kind: builtin
    name: file-read
---

Only report reproducible correctness, security, concurrency and
data-consistency problems. Do not modify files.
"""

_CODE_REVIEW_SKILL = """\
---
name: code-review
description: Review a change for correctness and security problems.
---

# Code Review

Read the diff, then report only reproducible problems.
Do not modify files.
"""

_SKILL_CREATOR_SKILL = """\
---
name: skill-creator
description: Create, improve and evaluate Agent Skills.
---

# Skill Creator

For every evaluation:

1. Run one Subagent with the candidate Skill.
2. Run one baseline Subagent without the Skill.
3. Use `agents/grader.md` to grade assertions.
4. Use `agents/comparator.md` for blind comparison.
5. Use `agents/analyzer.md` for final analysis.

Only pass explicitly required files and task context to each Subagent.
"""

_GRADER_MD = """\
---
name: grader
description: Grade an evaluation result.

tools:
  - kind: builtin
    name: file-read

timeout_seconds: 120
max_depth: 0
---

Evaluate each assertion. Return pass/fail, evidence and confidence.
"""

_COMPARATOR_MD = """\
---
name: comparator
description: Blind-compare two evaluation outputs.
---

Compare the two outputs and pick the stronger one, with reasons.
"""

_ANALYZER_MD = """\
---
name: analyzer
description: Final analysis across evaluations.
---

Aggregate the graded evaluations into a single recommendation.
"""

_GITHUB_MCP_DISABLED = """\
name: github

transport: stdio

command:
  - npx
  - -y
  - "@modelcontextprotocol/server-github"

env:
  GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"

discovery_mode: strict
tool_prefix: true

enabled_tools:
  - search_code
  - get_file_contents
"""

# (relative path, content) pairs, written in order.
_SCAFFOLD: "tuple[tuple[str, str], ...]" = (
    ("config.yaml", _CONFIG_YAML),
    ("agents/default.md", _DEFAULT_AGENT_MD),
    ("agents/reviewer.md", _REVIEWER_AGENT_MD),
    ("skills/code-review/SKILL.md", _CODE_REVIEW_SKILL),
    ("skills/skill-creator/SKILL.md", _SKILL_CREATOR_SKILL),
    ("skills/skill-creator/agents/grader.md", _GRADER_MD),
    ("skills/skill-creator/agents/comparator.md", _COMPARATOR_MD),
    ("skills/skill-creator/agents/analyzer.md", _ANALYZER_MD),
    ("mcp/github.yaml.disabled", _GITHUB_MCP_DISABLED),
)


def _write_if_missing(path: Path, content: str) -> bool:
    """Write ``content`` to ``path`` only if it does not already exist; create
    parent directories. Returns True if a new file was created."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def initialize_project(path: "Path | None") -> int:
    """Scaffold ``.linktools/`` under ``path`` (default: cwd)."""
    logger = environ.logger
    root = Path(path) if path else Path.cwd()
    config_root = root / ".linktools"
    created = 0
    skipped = 0
    for relative, content in _SCAFFOLD:
        if _write_if_missing(config_root / relative, content):
            created += 1
        else:
            skipped += 1
    logger.info(
        f"project initialized at {config_root} ({created} created, {skipped} kept)"
    )
    return 0
