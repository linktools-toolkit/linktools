#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Project discovery/config tests."""

import tempfile
import unittest
from pathlib import Path

from linktools.ai_cli.project import (
    ProjectConfigError,
    find_project_root,
    load_project,
    project_hash,
)


def _write_config(root: Path, body: str) -> None:
    (root / ".linktools").mkdir(parents=True, exist_ok=True)
    (root / ".linktools" / "config.yaml").write_text(body, encoding="utf-8")


_DEFAULT_CONFIG = """\
version: 1
default_agent: default
default_session: main
subagents:
  max_depth: 3
  max_concurrency: 4
  default_timeout_seconds: 120
mcp:
  allow_wildcard: false
"""


class TestFindProjectRoot(unittest.TestCase):
    def test_finds_config_in_current_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_config(root, _DEFAULT_CONFIG)
            self.assertEqual(find_project_root(root), root.resolve())

    def test_finds_config_from_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_config(root, _DEFAULT_CONFIG)
            deep = root / "a" / "b" / "c"
            deep.mkdir(parents=True)
            self.assertEqual(find_project_root(deep), root.resolve())

    def test_missing_config_defaults_to_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            # No config.yaml — find_project_root returns the start path, not an error
            self.assertEqual(find_project_root(Path(tmp)), Path(tmp).resolve())


class TestProjectHash(unittest.IsolatedAsyncioTestCase):
    def test_hash_is_stable_and_distinct(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            ha1 = project_hash(Path(a))
            ha2 = project_hash(Path(a))
            hb = project_hash(Path(b))
            self.assertEqual(ha1, ha2)
            self.assertNotEqual(ha1, hb)
            self.assertEqual(len(ha1), 16)


class TestLoadProject(unittest.TestCase):
    def _load(self, root: Path, data_root: Path):
        return load_project(data_root=data_root, start=root)

    def test_loads_valid_project(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as data,
        ):
            root = Path(tmp)
            _write_config(root, _DEFAULT_CONFIG)
            project = self._load(root, Path(data))
            self.assertEqual(project.default_agent, "default")
            self.assertEqual(project.default_session, "main")
            self.assertEqual(project.subagent_max_depth, 3)
            self.assertEqual(project.subagent_max_concurrency, 4)
            self.assertEqual(project.subagent_timeout_seconds, 120)
            self.assertFalse(project.allow_mcp_wildcard)
            self.assertEqual(project.agents_root, project.config_root / "agents")
            self.assertEqual(project.skills_root, project.config_root / "skills")

    def test_loads_without_config_uses_defaults(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as data,
        ):
            root = Path(tmp)
            # No config.yaml — load_project uses defaults
            project = self._load(root, Path(data))
            self.assertEqual(project.default_agent, "default")
            self.assertEqual(project.default_session, "main")
            self.assertEqual(project.subagent_max_depth, 3)
            self.assertEqual(project.root, root.resolve())

    def test_invalid_version_raises(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as data,
        ):
            root = Path(tmp)
            _write_config(root, "version: 2\ndefault_agent: x\n")
            with self.assertRaises(ProjectConfigError):
                self._load(root, Path(data))

    def test_non_mapping_raises(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as data,
        ):
            root = Path(tmp)
            _write_config(root, "- a\n- b\n")
            with self.assertRaises(ProjectConfigError):
                self._load(root, Path(data))

    def test_blank_agent_raises(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            tempfile.TemporaryDirectory() as data,
        ):
            root = Path(tmp)
            _write_config(root, "version: 1\ndefault_agent: ' '\ndefault_session: m\n")
            with self.assertRaises(ProjectConfigError):
                self._load(root, Path(data))


class TestProjectStateIsolation(unittest.TestCase):
    def test_two_projects_get_distinct_state_roots(self):
        with (
            tempfile.TemporaryDirectory() as a,
            tempfile.TemporaryDirectory() as b,
            tempfile.TemporaryDirectory() as data,
        ):
            ra, rb, data_root = Path(a), Path(b), Path(data)
            _write_config(ra, _DEFAULT_CONFIG)
            _write_config(rb, _DEFAULT_CONFIG)
            pa = load_project(data_root=data_root, start=ra)
            pb = load_project(data_root=data_root, start=rb)
            self.assertNotEqual(pa.state_root, pb.state_root)
            self.assertIn("projects", str(pa.state_root))
            # The isolating dir name is the per-project hash, so it differs too.
            self.assertNotEqual(pa.state_root.name, pb.state_root.name)


if __name__ == "__main__":
    unittest.main()
