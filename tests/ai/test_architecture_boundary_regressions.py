#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Architecture gate escape-path regressions."""

from pathlib import Path

from scripts.check.ai.architecture import ArchitecturePolicyChecker


def _source_tree(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "linktools" / "ai"
    root.mkdir(parents=True)
    (root / "__init__.py").write_text("__all__ = []\n", encoding="utf-8")
    for relative, source in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        package = path.parent
        while package != root:
            init = package / "__init__.py"
            if not init.exists():
                init.write_text("__all__ = []\n", encoding="utf-8")
            package = package.parent
        path.write_text(source, encoding="utf-8")
    return root


def test_external_alias_from_linktools_cannot_bypass_public_surface(tmp_path: Path) -> None:
    source_root = _source_tree(
        tmp_path,
        {
            "a/__init__.py": "Public = object()\nHidden = object()\n__all__ = ['Public']\n",
        },
    )
    external = tmp_path / "external" / "linktools"
    consumer = external / "commands" / "ai.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text(
        "from linktools import ai as public_ai\nvalue = public_ai.a.Hidden\n",
        encoding="utf-8",
    )

    errors = ArchitecturePolicyChecker().check(
        source_root,
        external_roots=(external,),
    ).errors

    assert any("cross-owner non-exported symbol access" in error for error in errors)


def test_external_import_linktools_attribute_path_is_scanned(tmp_path: Path) -> None:
    source_root = _source_tree(
        tmp_path,
        {
            "a/__init__.py": "Public = object()\nHidden = object()\n__all__ = ['Public']\n",
        },
    )
    external = tmp_path / "external" / "linktools"
    consumer = external / "commands" / "ai.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text(
        "import linktools\nvalue = linktools.ai.a.Hidden\n",
        encoding="utf-8",
    )

    errors = ArchitecturePolicyChecker().check(
        source_root,
        external_roots=(external,),
    ).errors

    assert any("cross-owner non-exported symbol access" in error for error in errors)


def test_external_parent_package_alias_is_resolved(tmp_path: Path) -> None:
    source_root = _source_tree(
        tmp_path,
        {
            "a/__init__.py": "Public = object()\nHidden = object()\n__all__ = ['Public']\n",
        },
    )
    external = tmp_path / "external" / "linktools"
    consumer = external / "commands" / "ai.py"
    consumer.parent.mkdir(parents=True)
    consumer.write_text(
        "import linktools as lt\nvalue = lt.ai.a.Hidden\n",
        encoding="utf-8",
    )

    errors = ArchitecturePolicyChecker().check(
        source_root,
        external_roots=(external,),
    ).errors

    assert any("cross-owner non-exported symbol access" in error for error in errors)


def test_all_direct_mutations_are_not_static_export_contracts(tmp_path: Path) -> None:
    variants = (
        "__all__ = ['Public']\n__all__.append('Hidden')\n",
        "__all__ = ['Public']\n__all__.clear()\n",
        "__all__ = ['Public']\n__all__[0] = 'Hidden'\n",
        "__all__ = ['Public']\ndel __all__\n",
    )

    for index, mutation in enumerate(variants):
        root = _source_tree(
            tmp_path / str(index),
            {"a/__init__.py": "Public = object()\nHidden = object()\n" + mutation},
        )
        errors = ArchitecturePolicyChecker().check(root).errors
        assert any("__all__ must be one static string sequence" in error for error in errors)


def test_all_readonly_methods_do_not_invalidate_static_contract(tmp_path: Path) -> None:
    root = _source_tree(
        tmp_path,
        {
            "a/__init__.py": (
                "Public = object()\n"
                "__all__ = ['Public']\n"
                "count = __all__.count('Public')\n"
                "position = __all__.index('Public')\n"
            ),
        },
    )

    assert ArchitecturePolicyChecker().check(root).passed
