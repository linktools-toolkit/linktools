#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Skill view helpers and utility functions."""

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import SkillSpec


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_csv(value: Any) -> "list[str]":
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [t.strip() for t in str(value).split(",") if t.strip()]


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------

def view_skill(
    skills: "list[SkillSpec]",
    skill_id: str,
    file_path: "str | None" = None,
) -> "dict[str, object]":
    """Load a skill's instructions and linked file listing, or a single file."""
    skill = _find_skill(skills, skill_id)
    if skill is None:
        return {"success": False, "error": f"skill not available: {skill_id!r}"}

    if file_path is not None:
        return _view_skill_file(skill, skill_id, file_path)

    main_file = skill.path
    if main_file.is_file():
        from ..support.config import load_markdown_file as _load_markdown_file
        metadata, body = _load_markdown_file(main_file)
    else:
        metadata, body = skill.metadata, skill.instructions
    content = _resolve_paths(body, skill.base_dir) if skill.base_dir else body
    linked_files = _list_linked_files(skill.base_dir) if skill.base_dir and skill.base_dir.is_dir() else {}
    skill_path = skill.name + "/" + main_file.name

    return {
        "success": True,
        "skill_id": skill.name,
        "name": skill.name,
        "description": skill.description,
        "tags": _parse_csv(metadata.get("tags", "")),
        "related_skills": _parse_csv(metadata.get("related_skills", "")),
        "content": content,
        "path": skill_path,
        "skill_dir": str(skill.base_dir) if skill.base_dir else None,
        "linked_files": linked_files or None,
        "usage_hint": (
            "All paths in `content` and `linked_files` are relative to `skill_dir`. "
            "To fetch a file's content, call the same skill view tool again with file_path set to the relative path, "
            "e.g. file_path='references/api.md'. "
            "To execute a skill script, construct its absolute path as `skill_dir + '/' + relative_path` and call bash. "
            "Do not use read_file/list_dir for skill files."
        ) if linked_files else None,
    }


def view_available_skills(
    skills: "list[SkillSpec]",
    file_path: "str | None" = None,
) -> "dict[str, object]":
    if not skills:
        return {"success": False, "error": "no skills available"}
    if file_path is None:
        return {
            "success": True,
            "skills": [view_skill(skills, skill.name) for skill in skills],
        }
    matches = [
        skill
        for skill in skills
        if skill.base_dir is not None and _resolve_skill_file_if_exists(skill.base_dir, file_path) is not None
    ]
    if not matches:
        return {
            "success": False,
            "error": f"File {file_path!r} not found in available skills.",
            "available_skills": [skill.name for skill in skills],
        }
    if len(matches) > 1:
        return {
            "success": False,
            "error": f"File path {file_path!r} is ambiguous across available skills.",
            "matching_skills": [skill.name for skill in matches],
        }
    return view_skill(skills, matches[0].name, file_path=file_path)


# ---------------------------------------------------------------------------
# Public helpers consumed by executor
# ---------------------------------------------------------------------------

def skill_summaries(skills: "list[SkillSpec]") -> "list[dict[str, object]]":
    """Minimal summary for prompt injection — no SKILL.md body."""
    return [
        {
            "skill_id": skill.name,
            "name": skill.name,
            "category": skill.category,
            "description": skill.description,
        }
        for skill in skills
    ]



# ---------------------------------------------------------------------------
# Linked file listing
# ---------------------------------------------------------------------------

_EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo", ".pyd", ".class", ".o", ".obj", ".so", ".dll", ".exe"})
_EXCLUDED_NAMES   = frozenset({"__pycache__", ".git", ".DS_Store", "Thumbs.db"})


def _list_linked_files(base_dir: Path) -> "dict[str, list[str]]":
    """Return skill resource file names grouped by convention category (no content)."""
    result: "dict[str, list[str]]" = {}

    def _relpaths(directory: Path) -> "list[str]":
        return sorted(
            f.relative_to(base_dir).as_posix()
            for f in directory.rglob("*")
            if f.is_file()
            and not any(part.startswith(".") or part in _EXCLUDED_NAMES for part in f.parts)
            and f.suffix not in _EXCLUDED_SUFFIXES
        )

    for category in ("references", "templates", "assets", "scripts"):
        d = base_dir / category
        if d.is_dir():
            files = _relpaths(d)
            if files:
                result[category] = files
    return result


# ---------------------------------------------------------------------------
# Single-file view
# ---------------------------------------------------------------------------

def _view_skill_file(skill: "SkillSpec", skill_id: str, file_path: str) -> "dict[str, object]":
    if skill.base_dir is None:
        return {"success": False, "error": f"skill {skill_id!r} has no filesystem-backed files"}
    try:
        abs_path = _resolve_skill_file(skill.base_dir, file_path)
    except ValueError as exc:
        return {"success": False, "error": str(exc), "hint": "Use a relative path within the skill directory"}

    if not abs_path.exists():
        return {
            "success": False,
            "error": f"File {file_path!r} not found in skill {skill_id!r}.",
            "available_files": _list_linked_files(skill.base_dir),
            "hint": "Use one of the available_files paths listed above",
        }

    try:
        content = abs_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {
            "success": True,
            "skill_id": skill_id,
            "file": file_path,
            "content": f"[Binary file: {abs_path.name}, size: {abs_path.stat().st_size} bytes]",
            "is_binary": True,
        }
    except OSError as exc:
        return {"success": False, "error": f"cannot read file: {exc}"}

    return {
        "success": True,
        "skill_id": skill_id,
        "file": file_path,
        "content": content,
        "file_type": abs_path.suffix,
    }


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def _resolve_skill_file(base_dir: Path, rel_path: str) -> Path:
    clean = rel_path.strip()
    if not clean:
        raise ValueError("file_path must not be empty")
    p = Path(clean)
    if p.is_absolute():
        raise ValueError(f"absolute paths are not allowed: {clean!r}")
    # Covers both "." and ".." — blocks hidden files and parent-dir traversal.
    if any(part.startswith(".") for part in p.parts):
        raise ValueError(f"hidden-file access is not allowed: {clean!r}")
    # Check containment on the logical (pre-symlink) path so that symlinks inside
    # base_dir pointing to L2 cache entries outside it are not falsely rejected.
    if not _is_under_logical(base_dir / p, base_dir):
        raise ValueError(f"path traversal detected: {clean!r}")
    return (base_dir / p).resolve()


def _resolve_skill_file_if_exists(base_dir: Path, rel_path: str) -> "Path | None":
    try:
        path = _resolve_skill_file(base_dir, rel_path)
    except ValueError:
        return None
    return path if path.exists() else None


# ---------------------------------------------------------------------------
# Path resolution in SKILL.md body
# ---------------------------------------------------------------------------

def _resolve_paths(content: str, base_dir: Path) -> str:
    return content.replace("${SKILL_DIR}", str(base_dir.resolve()))


# ---------------------------------------------------------------------------
# Low-level utilities
# ---------------------------------------------------------------------------

def _find_skill(skills: "list[SkillSpec]", skill_id: str) -> "SkillSpec | None":
    wanted = skill_id.strip()
    for skill in skills:
        if skill.name == wanted:
            return skill
    return None


def _is_under(path: Path, root: Path) -> bool:
    try:
        _ = path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_under_logical(path: Path, root: Path) -> bool:
    """Containment check on the logical path without following symlinks."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
