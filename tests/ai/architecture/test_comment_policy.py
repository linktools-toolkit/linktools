#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comment-policy scanner: comments and docstrings across the repo must not
carry external-section refs, review-item tags, task numbers, or process/state
markers (§11.2). Comments exist to convey invariants and non-obvious behavior
-- not to locate a clause in a design doc or narrate history.

Implements the §11.5 contract (tokenize COMMENT + ast docstring) plus a
companion config-comment guard (§11.4 #2 whole-repo) and a precise
well-formedness guard that locks the cleanup against the orphan-punctuation
classes that have unambiguous, zero-legit-baseline forms.

Two marker scopes:
* REPO_WIDE_PATTERNS -- unambiguous markers enforced over every Python tree.
* SRC_STRICT_PATTERNS -- the bare word "plan", enforced over shipped src only
  (tests legitimately use it as a domain noun: ExecutionPlanner, "a plan must
  never execute a write").

This file is in _EXEMPT so it may name its own patterns in comments without
flagging itself."""

from __future__ import annotations

import ast
import tokenize
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AI_SRC = _REPO_ROOT / "linktools-ai" / "src" / "linktools" / "ai"
_SHIPPED_SRC = _REPO_ROOT / "linktools-ai" / "src" / "linktools"

_SCAN_TREES = (
    _REPO_ROOT / "linktools-ai" / "src" / "linktools" / "ai",
    _REPO_ROOT / "linktools-ai" / "src" / "linktools" / "ai_cli",
    _REPO_ROOT / "linktools-ai" / "src" / "linktools" / "capabilities",
    _REPO_ROOT / "linktools-ai" / "src" / "linktools" / "commands",
    _REPO_ROOT / "linktools-ai" / "scripts",
    _REPO_ROOT / "linktools-ai-testing" / "src",
    _REPO_ROOT / "linktools-ai" / "conformance",
    _REPO_ROOT / "tests",
)

# Files exempt from the scan. This scanner holds its own patterns as data;
# migrate.py names old table/directory identifiers it is contracted to move.
_EXEMPT = {
    Path("tests") / "ai" / "architecture" / "test_comment_policy.py",
    Path("linktools-ai") / "src" / "linktools" / "ai" / "storage" / "migrate.py",
}

# Unambiguous forbidden markers (§11.2). Applied to every scanned tree.
REPO_WIDE_PATTERNS: "tuple[str, ...]" = (
    r"[Pp]lan\s*§",
    r"[Pp]lan\s+sections?",
    r"[Pp]lan\s+ops?",
    r"[Pp]lan\s+steps?",
    r"[Pp]lan\s+phase",
    r"[Pp]lan\s+\d+(?:\.\d+)*(?:\s*[/,]\s*\d+(?:\.\d+)*)*",
    r"spec\s*§",
    r"design\s*§",
    r"fix-plan",
    r"plan-mandated",
    r"\bthe\s+[Pp]lan\b",
    r"\bthis\s+[Pp]lan\b",
    r"\b[Pp]lan's\b",
    r"review\s+item",
    r"per\s+review",
    r"see\s+design",
    r"按方案",
    r"设计章节",
    r"实现步骤",
    r"\bR-?\d+[a-z]?\b",
    r"\bSEC-\d+\b",
    r"\bAC-\d+\b",
    r"\bWP-\d+\b",
    r"\bRF-\d+\b",
    r"\bBUG-\d+\b",
    r"\bA-\d{2,}\b",
    r"\bB-\d{2,}\b",
    r"\bC-\d{2,}\b",
    r"\bR\d+[a-z]?-R\d+\b",
    r"production-hardening\s+plan",
    r"hardening\s+plan",
    r"not\s+yet\s+wired",
    r"backward\s+compatibility",
    r"frozen\s+constructor",
    r"later\s+work",
    r"\blater\s+phase",
    r"\bellipses?\b",
    r"本轮修改",
    r"历史上",
    r"此前为了",
    r"聊天记录",
    r"\bReviewer\b",
    r"\bPR\b",
    r"\bIssue\b",
    r"§\s*\d",
    r"\bPhase\s+\d",
    r"兼容旧",
    r"currently\s+rejects",
    r"SQLite\s+UPSERT",
)

# §11.5 additions scoped to linktools-ai only (src/linktools/ai + ai_cli +
# tests/ai). Other sub-packages (cntr, core, mobile) carry their own review /
# task codes from their own processes; this plan's comment policy does not
# govern them. The ``review`` forms target review-PROCESS refs (numbered
# review rounds, current-/final-review) -- NOT legit project feature names
# like "code-review skill" or "reviewer agent", which this scanner must spare.
AI_SCOPED_PATTERNS: "tuple[str, ...]" = (
    r"\bP[0-9]+-[0-9]+\b",
    r"\bG[0-9]+\b",
    r"\breview\d+\b",
    r"\bcurrent-review\b",
    r"\bfinal-review\b",
    r"\breview\s+(?:contract|caught|round)\b",
    r"方案",
    r"实施步骤",
    r"本轮",
    r"后续处理",
    r"compatibility shim",
)

_AI_SCOPED_TREES = (
    _AI_SRC,
    _REPO_ROOT / "linktools-ai" / "src" / "linktools" / "ai_cli",
    _REPO_ROOT / "tests" / "ai",
)

SRC_STRICT_PATTERNS: "tuple[str, ...]" = (
    r"\b[Pp]lan\b",
)

# Precise well-formedness guards: orphan-punctuation forms that have ZERO
# legitimate baseline matches, so they unambiguously signal scrub damage. (The
# broader classes -- empty parens, em-dash, mid-line period -- all have legit
# baseline prose, so they are deliberately NOT enforced here.)
WELL_FORMED_PATTERNS: "tuple[str, ...]" = (
    r"-{2,}\s*:\s+",  # '# --- :' / docstring '--- :' heading orphan colon (space after, excludes Sphinx ':class:')
    r", \)",          # ', )' comma-close orphan (space before, excludes tuple-literal ',)')
)


def _py_files() -> "list[tuple[Path, bool]]":
    out: "list[tuple[Path, bool]]" = []
    seen: set[Path] = set()
    for tree in _SCAN_TREES:
        for p in tree.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            if p in seen:
                continue
            try:
                rel = p.relative_to(_REPO_ROOT)
            except ValueError:
                continue
            if rel in _EXEMPT:
                continue
            seen.add(p)
            out.append((p, _SHIPPED_SRC in p.parents))
    return out


def _comments(path: Path) -> "list[tuple[int, str]]":
    raw: "list[tuple[int, str]]" = []
    try:
        with open(path, "rb") as f:
            for tok in tokenize.tokenize(f.readline):
                if tok.type == tokenize.COMMENT:
                    raw.append((tok.start[0], tok.string))
    except (tokenize.TokenizeError, SyntaxError, OSError):
        return raw
    out: "list[tuple[int, str]]" = []
    for lineno, s in raw:
        text = s.lstrip("#").strip()
        if out and out[-1][0] == lineno - 1:
            prev_lineno, prev_text = out[-1]
            out[-1] = (prev_lineno, f"{prev_text} {text}")
        else:
            out.append((lineno, text))
    return out


def _docstrings(path: Path) -> "list[tuple[int, str]]":
    out: "list[tuple[int, str]]" = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                out.append((getattr(node, "lineno", 1), doc))
    return out


def _scan(patterns: "tuple[str, ...]", src_only: bool) -> "list[tuple[str, int, str, str]]":
    import re

    compiled = [re.compile(p) for p in patterns]
    hits: "list[tuple[str, int, str, str]]" = []
    for path, is_src in _py_files():
        if src_only and not is_src:
            continue
        rel = str(path.relative_to(_REPO_ROOT))
        for lineno, text in _comments(path) + _docstrings(path):
            for rx in compiled:
                m = rx.search(text)
                if m:
                    hits.append((rel, lineno, m.group(0), rx.pattern))
    return hits


def _scan_trees(
    trees: "tuple[Path, ...]", patterns: "tuple[str, ...]"
) -> "list[tuple[str, int, str, str]]":
    import re

    compiled = [re.compile(p) for p in patterns]
    hits: "list[tuple[str, int, str, str]]" = []
    seen: set[Path] = set()
    for tree in trees:
        if not tree.exists():
            continue
        for path in tree.rglob("*.py"):
            if "__pycache__" in path.parts or path in seen:
                continue
            seen.add(path)
            rel = str(path.relative_to(_REPO_ROOT))
            if rel in {str(p) for p in _EXEMPT}:
                continue
            for lineno, text in _comments(path) + _docstrings(path):
                for rx in compiled:
                    m = rx.search(text)
                    if m:
                        hits.append((rel, lineno, m.group(0), rx.pattern))
    return hits


def test_no_forbidden_comment_or_docstring_markers() -> None:
    hits = _scan(REPO_WIDE_PATTERNS, src_only=False)
    if hits:
        rendered = "\n".join(
            f"  {f}:{line}: {matched!r} (pattern {pat!r})"
            for f, line, matched, pat in hits[:50]
        )
        pytest.fail(
            f"forbidden comment/docstring markers found ({len(hits)}):\n{rendered}"
        )


def test_no_ai_scoped_forbidden_markers() -> None:
    # §11.5 markers scoped to linktools-ai only (other sub-packages carry
    # their own review/task codes this plan does not govern).
    hits = _scan_trees(_AI_SCOPED_TREES, AI_SCOPED_PATTERNS)
    if hits:
        rendered = "\n".join(
            f"  {f}:{line}: {matched!r} (pattern {pat!r})"
            for f, line, matched, pat in hits[:50]
        )
        pytest.fail(
            f"forbidden linktools-ai comment/docstring markers ({len(hits)}):\n{rendered}"
        )


def test_src_has_no_bare_plan_word() -> None:
    hits = _scan(SRC_STRICT_PATTERNS, src_only=True)
    if hits:
        rendered = "\n".join(
            f"  {f}:{line}: {matched!r} (pattern {pat!r})"
            for f, line, matched, pat in hits[:50]
        )
        pytest.fail(
            f'bare "plan" word in shipped src ({len(hits)}):\n{rendered}'
        )


def test_no_orphan_punctuation() -> None:
    # Precise well-formedness guard (zero-legit-baseline forms only). Locks the
    # scrub against the orphan-punctuation classes that unambiguously signal
    # damage; broader classes are excluded because they have legit prose matches.
    hits = _scan(WELL_FORMED_PATTERNS, src_only=False)
    if hits:
        rendered = "\n".join(
            f"  {f}:{line}: {matched!r} (pattern {pat!r})"
            for f, line, matched, pat in hits[:50]
        )
        pytest.fail(
            f"orphan punctuation from scrub damage ({len(hits)}):\n{rendered}"
        )


def test_no_forbidden_markers_in_config_comments() -> None:
    # §11.4 #2 whole-repo: non-Python config comments too.
    import re

    compiled = [re.compile(p) for p in REPO_WIDE_PATTERNS]
    hits: "list[tuple[str, int, str]]" = []
    for tree in _SCAN_TREES:
        for ext in ("*.toml", "*.yaml", "*.yml", "*.cfg", "*.ini"):
            for p in tree.rglob(ext):
                if "__pycache__" in p.parts:
                    continue
                try:
                    rel = str(p.relative_to(_REPO_ROOT))
                except ValueError:
                    continue
                for lineno, line in enumerate(
                    p.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    stripped = line.lstrip()
                    if not stripped.startswith("#"):
                        continue
                    for rx in compiled:
                        m = rx.search(stripped)
                        if m:
                            hits.append((rel, lineno, m.group(0)))
                            break
    if hits:
        rendered = "\n".join(
            f"  {f}:{line}: {matched!r}" for f, line, matched in hits[:50]
        )
        pytest.fail(
            f"forbidden markers in config comments ({len(hits)}):\n{rendered}"
        )


def test_string_literals_are_not_scanned() -> None:
    import io
    import re
    import tokenize as tz

    sample = 'x = "see design fallback"\n# real comment\n'
    compiled = [re.compile(p) for p in REPO_WIDE_PATTERNS]
    comments = [
        t.string
        for t in tz.tokenize(io.BytesIO(sample.encode()).readline)
        if t.type == tz.COMMENT
    ]
    flagged = any(rx.search(c) for c in comments for rx in compiled)
    assert not flagged, "scanner must not flag forbidden markers inside string literals"
