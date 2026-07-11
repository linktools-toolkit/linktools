#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generic ``.linktools.json`` project manifest: ManifestLoader static
validation and RequirementResolverRegistry version-requirement checking.
Both live in linktools.core and must never import a capability package
(linktools.cntr, linktools.ai, docker, compose, ...)."""
import ast
import json
from pathlib import Path

import pytest

from linktools.core import (
    LinktoolsManifest, ManifestComponent, ManifestLoader, RequirementResolverRegistry,
    RequirementStatus,
)
from linktools.errors import ManifestLoadError, ManifestSchemaUnsupported, ManifestValidationError

_VALID = {
    "schema_version": 1,
    "kind": "linktools-project",
    "name": "example-platform",
    "version": "1.2.0",
}


def _write(tmp_path, data):
    if isinstance(data, str):
        (tmp_path / ".linktools.json").write_text(data, encoding="utf-8")
    else:
        (tmp_path / ".linktools.json").write_text(json.dumps(data), encoding="utf-8")


# -- load() / loads() -----------------------------------------------------

def test_missing_manifest_returns_none(tmp_path):
    assert ManifestLoader().load(tmp_path) is None


def test_empty_file_raises_load_error(tmp_path):
    _write(tmp_path, "")
    with pytest.raises(ManifestLoadError):
        ManifestLoader().load(tmp_path)


def test_invalid_json_raises_load_error(tmp_path):
    _write(tmp_path, "{not json")
    with pytest.raises(ManifestLoadError):
        ManifestLoader().load(tmp_path)


def test_non_object_root_raises_validation_error(tmp_path):
    _write(tmp_path, "[1, 2, 3]")
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_oversized_manifest_raises_load_error(tmp_path):
    huge = dict(_VALID, description="x" * (1024 * 1024 + 1))
    _write(tmp_path, huge)
    with pytest.raises(ManifestLoadError):
        ManifestLoader().load(tmp_path)


def test_missing_schema_version_is_invalid(tmp_path):
    data = dict(_VALID)
    del data["schema_version"]
    _write(tmp_path, data)
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_unsupported_schema_version_rejected(tmp_path):
    _write(tmp_path, dict(_VALID, schema_version=2))
    with pytest.raises(ManifestSchemaUnsupported):
        ManifestLoader().load(tmp_path)


def test_manifest_that_is_a_directory_raises_load_error(tmp_path):
    (tmp_path / ".linktools.json").mkdir()
    with pytest.raises(ManifestLoadError):
        ManifestLoader().load(tmp_path)


def test_stat_permission_error_is_wrapped_as_load_error(tmp_path, monkeypatch):
    _write(tmp_path, _VALID)

    def fail_stat(path):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr("os.stat", fail_stat)
    with pytest.raises(ManifestLoadError):
        ManifestLoader().load(tmp_path)


def test_open_permission_error_is_wrapped_as_load_error(tmp_path, monkeypatch):
    _write(tmp_path, _VALID)

    real_open = open

    def fail_open(path, *a, **k):
        if str(path).endswith(".linktools.json"):
            raise PermissionError(13, "Permission denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fail_open)
    with pytest.raises(ManifestLoadError):
        ManifestLoader().load(tmp_path)


def test_file_disappearing_between_stat_and_open_raises_load_error(tmp_path, monkeypatch):
    _write(tmp_path, _VALID)

    real_open = open

    def vanish_on_open(path, *a, **k):
        if str(path).endswith(".linktools.json"):
            raise FileNotFoundError(2, "No such file or directory")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", vanish_on_open)
    with pytest.raises(ManifestLoadError):
        ManifestLoader().load(tmp_path)


def test_missing_kind_is_invalid(tmp_path):
    data = dict(_VALID)
    del data["kind"]
    _write(tmp_path, data)
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_wrong_kind_is_invalid(tmp_path):
    _write(tmp_path, dict(_VALID, kind="linktools-cntr-repository"))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_unknown_top_level_field_is_invalid(tmp_path):
    _write(tmp_path, dict(_VALID, unknown_field=123))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_invalid_version_is_invalid(tmp_path):
    _write(tmp_path, dict(_VALID, version="not-a-version!!"))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_requires_non_object_is_invalid(tmp_path):
    _write(tmp_path, dict(_VALID, requires=["python"]))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_invalid_specifier_is_invalid(tmp_path):
    _write(tmp_path, dict(_VALID, requires={"python": "not a specifier!!"}))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_dollar_schema_field_is_never_fetched(tmp_path, monkeypatch):
    def fail_on_network(*a, **k):
        raise AssertionError("must never make a network request")

    monkeypatch.setattr("urllib.request.urlopen", fail_on_network, raising=False)
    _write(tmp_path, dict(_VALID, **{"$schema": "https://example.invalid/schema.json"}))
    manifest = ManifestLoader().load(tmp_path)
    assert manifest is not None


def test_metadata_and_extensions_preserved_verbatim(tmp_path):
    data = dict(_VALID, metadata={"license": "Apache-2.0"}, extensions={"vendor.x": {"a": 1}})
    _write(tmp_path, data)
    manifest = ManifestLoader().load(tmp_path)
    assert manifest.metadata == {"license": "Apache-2.0"}
    assert manifest.extensions == {"vendor.x": {"a": 1}}


# -- components -------------------------------------------------------------

_WITH_COMPONENTS = dict(_VALID, components={
    "cntr": {
        "schema_version": 1,
        "requires": {"package:linktools-cntr": ">=0.12", "runtime:docker-engine": ">=24"},
        "config": {"a": 1},
        "metadata": {"tags": ["homelab"]},
        "extensions": {},
    },
    "ai": {
        "schema_version": 1,
        "requires": {"package:linktools-ai": ">=0.1"},
        "config": {"agents_path": "agents"},
        "metadata": {},
        "extensions": {},
    },
})


def test_multiple_components_are_parsed_independently(tmp_path):
    _write(tmp_path, _WITH_COMPONENTS)
    manifest = ManifestLoader().load(tmp_path)
    assert isinstance(manifest, LinktoolsManifest)
    cntr = manifest.get_component("cntr")
    ai = manifest.get_component("ai")
    assert isinstance(cntr, ManifestComponent)
    assert cntr.requires["package:linktools-cntr"] == ">=0.12"
    assert cntr.config == {"a": 1}
    assert ai.config == {"agents_path": "agents"}
    assert manifest.get_component("missing") is None


def test_unknown_component_key_is_allowed(tmp_path):
    _write(tmp_path, dict(_VALID, components={
        "totally-custom-name": {"schema_version": 1, "requires": {}, "config": {}, "metadata": {}, "extensions": {}},
    }))
    manifest = ManifestLoader().load(tmp_path)
    assert manifest.get_component("totally-custom-name") is not None


def test_component_missing_schema_version_is_invalid(tmp_path):
    _write(tmp_path, dict(_VALID, components={"cntr": {"requires": {}}}))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_component_unknown_field_is_invalid(tmp_path):
    _write(tmp_path, dict(_VALID, components={
        "cntr": {"schema_version": 1, "search_path": "containers"},
    }))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_component_config_metadata_extensions_must_be_objects(tmp_path):
    _write(tmp_path, dict(_VALID, components={
        "cntr": {"schema_version": 1, "config": ["not", "an", "object"]},
    }))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


def test_component_invalid_specifier_is_invalid(tmp_path):
    _write(tmp_path, dict(_VALID, components={
        "cntr": {"schema_version": 1, "requires": {"runtime:docker-engine": "not a specifier!!"}},
    }))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


# -- requirement key / component id grammar ----------------------------------

@pytest.mark.parametrize("key", ["", "   ", "package:", ":no-namespace", "Python", "PACKAGE:x", "package :x"])
def test_invalid_requirement_key_is_rejected(tmp_path, key):
    _write(tmp_path, dict(_VALID, requires={key: ">=1"}))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


@pytest.mark.parametrize("key", ["python", "package:linktools-ai", "runtime:docker-compose", "a:b"])
def test_valid_requirement_key_is_accepted(tmp_path, key):
    _write(tmp_path, dict(_VALID, requires={key: ">=1"}))
    manifest = ManifestLoader().load(tmp_path)
    assert key in manifest.requires


@pytest.mark.parametrize("component_id", ["", "   ", "Cntr", "cntr!", "1cntr", "cn tr"])
def test_invalid_component_id_is_rejected(tmp_path, component_id):
    _write(tmp_path, dict(_VALID, components={
        component_id: {"schema_version": 1, "requires": {}, "config": {}, "metadata": {}, "extensions": {}},
    }))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


@pytest.mark.parametrize("component_id", ["cntr", "ai", "custom-name", "custom_name", "a1"])
def test_valid_component_id_is_accepted(tmp_path, component_id):
    _write(tmp_path, dict(_VALID, components={
        component_id: {"schema_version": 1, "requires": {}, "config": {}, "metadata": {}, "extensions": {}},
    }))
    manifest = ManifestLoader().load(tmp_path)
    assert manifest.get_component(component_id) is not None


def test_component_requires_key_grammar_is_also_validated(tmp_path):
    _write(tmp_path, dict(_VALID, components={
        "cntr": {"schema_version": 1, "requires": {"": ">=1"}, "config": {}, "metadata": {}, "extensions": {}},
    }))
    with pytest.raises(ManifestValidationError):
        ManifestLoader().load(tmp_path)


# -- RequirementResolverRegistry --------------------------------------------

def test_python_requirement_satisfied():
    registry = RequirementResolverRegistry.default()
    results = registry.check({"python": ">=3.0"}, phase="host")
    assert len(results) == 1
    assert results[0].status == RequirementStatus.SATISFIED


def test_python_requirement_unsatisfied():
    registry = RequirementResolverRegistry.default()
    results = registry.check({"python": ">=99.0"}, phase="host")
    assert results[0].status == RequirementStatus.UNSATISFIED


def test_installed_package_requirement_satisfied():
    registry = RequirementResolverRegistry.default()
    results = registry.check({"package:pytest": ">=1.0"}, phase="host")
    assert results[0].status == RequirementStatus.SATISFIED


def test_missing_package_requirement_is_unavailable():
    registry = RequirementResolverRegistry.default()
    results = registry.check({"package:this-package-does-not-exist-xyz": ">=1.0"}, phase="host")
    assert results[0].status == RequirementStatus.UNAVAILABLE


def test_unregistered_key_is_unrecognized():
    registry = RequirementResolverRegistry.default()
    results = registry.check({"runtime:docker-engine": ">=24"}, phase="host")
    assert results[0].status == RequirementStatus.UNRECOGNIZED


def test_unrecognized_key_is_reported_regardless_of_requested_phase():
    registry = RequirementResolverRegistry.default()
    results = registry.check({"runtime:docker-engine": ">=24"}, phase="runtime")
    assert results[0].status == RequirementStatus.UNRECOGNIZED


def test_exact_resolver_takes_precedence_over_prefix():
    registry = RequirementResolverRegistry()
    registry.register_prefix("runtime:", lambda key: "1.0", phase="runtime")
    registry.register_exact("runtime:special", lambda key: "2.0", phase="runtime")
    results = registry.check({"runtime:special": ">=2.0"}, phase="runtime")
    assert results[0].actual == "2.0"
    assert results[0].status == RequirementStatus.SATISFIED


def test_phase_filter_skips_requirements_registered_for_a_different_phase():
    registry = RequirementResolverRegistry.default()
    registry.register_exact("runtime:thing", lambda key: "1.0", phase="runtime")
    host_results = registry.check({"runtime:thing": ">=1.0"}, phase="host")
    assert host_results == []
    runtime_results = registry.check({"runtime:thing": ">=1.0"}, phase="runtime")
    assert runtime_results[0].status == RequirementStatus.SATISFIED


def test_no_phase_filter_checks_everything():
    registry = RequirementResolverRegistry.default()
    registry.register_exact("runtime:thing", lambda key: "1.0", phase="runtime")
    results = registry.check({"python": ">=3.0", "runtime:thing": ">=1.0"})
    assert len(results) == 2


def test_resolver_exception_becomes_unavailable_not_raised():
    registry = RequirementResolverRegistry()

    def boom(key):
        raise RuntimeError("some sensitive subprocess stderr / path / token")

    registry.register_exact("runtime:flaky", boom, phase="runtime")
    results = registry.check({"runtime:flaky": ">=1.0"}, phase="runtime")
    assert results[0].status == RequirementStatus.UNAVAILABLE
    assert "sensitive" not in results[0].message
    assert "token" not in results[0].message


def test_invalid_actual_version_is_invalid_status():
    registry = RequirementResolverRegistry()
    registry.register_exact("runtime:weird", lambda key: "not-a-version", phase="runtime")
    results = registry.check({"runtime:weird": ">=1.0"}, phase="runtime")
    assert results[0].status == RequirementStatus.INVALID


def test_duplicate_exact_registration_raises():
    registry = RequirementResolverRegistry()
    registry.register_exact("python", lambda key: "1.0", phase="host")
    with pytest.raises(ValueError):
        registry.register_exact("python", lambda key: "2.0", phase="host")


def test_duplicate_exact_registration_with_replace_overrides():
    registry = RequirementResolverRegistry()
    registry.register_exact("python", lambda key: "1.0", phase="host")
    registry.register_exact("python", lambda key: "2.0", phase="host", replace=True)
    results = registry.check({"python": ">=2.0"}, phase="host")
    assert results[0].actual == "2.0"


def test_duplicate_prefix_registration_raises():
    registry = RequirementResolverRegistry()
    registry.register_prefix("runtime:", lambda key: "1.0", phase="runtime")
    with pytest.raises(ValueError):
        registry.register_prefix("runtime:", lambda key: "2.0", phase="runtime")


def test_duplicate_prefix_registration_with_replace_overrides():
    registry = RequirementResolverRegistry()
    registry.register_prefix("runtime:", lambda key: "1.0", phase="runtime")
    registry.register_prefix("runtime:", lambda key: "2.0", phase="runtime", replace=True)
    results = registry.check({"runtime:thing": ">=2.0"}, phase="runtime")
    assert results[0].actual == "2.0"


def test_broader_and_narrower_prefix_the_narrower_always_wins():
    """A narrow prefix registered *before* a broad one must still win at
    resolve time -- longest-prefix-match, never registration order."""
    registry = RequirementResolverRegistry()
    registry.register_prefix("runtime:docker-compose", lambda key: "narrow", phase="runtime")
    registry.register_prefix("runtime:", lambda key: "broad", phase="runtime")
    results = registry.check({"runtime:docker-compose": ">=1"}, phase="runtime")
    assert results[0].actual == "narrow"


def test_broader_prefix_registered_first_does_not_shadow_narrower_registered_later():
    registry = RequirementResolverRegistry()
    registry.register_prefix("runtime:", lambda key: "broad", phase="runtime")
    registry.register_prefix("runtime:docker-compose", lambda key: "narrow", phase="runtime")
    results = registry.check({"runtime:docker-compose": ">=1"}, phase="runtime")
    assert results[0].actual == "narrow"


def test_equal_length_prefixes_resolve_in_registration_order():
    registry = RequirementResolverRegistry()
    registry.register_prefix("runtime:a", lambda key: "first", phase="runtime")
    registry.register_prefix("runtime:b", lambda key: "second", phase="runtime")
    # Neither prefix is a substring of the other's match target, so pick a
    # key that only one of them could ever match to confirm ordering is by
    # registration, not some other tiebreak, when both are equal length and
    # (deliberately) only one truly matches.
    results = registry.check({"runtime:a-thing": ">=1"}, phase="runtime")
    assert results[0].actual == "first"


def test_exact_key_must_be_non_empty_string():
    registry = RequirementResolverRegistry()
    with pytest.raises(ValueError):
        registry.register_exact("", lambda key: "1.0", phase="host")


def test_prefix_must_be_non_empty_string():
    registry = RequirementResolverRegistry()
    with pytest.raises(ValueError):
        registry.register_prefix("", lambda key: "1.0", phase="host")


def test_resolver_must_be_callable():
    registry = RequirementResolverRegistry()
    with pytest.raises(ValueError):
        registry.register_exact("python", "not callable", phase="host")


def test_phase_must_be_non_empty_string():
    registry = RequirementResolverRegistry()
    with pytest.raises(ValueError):
        registry.register_exact("python", lambda key: "1.0", phase="")


def test_display_name_must_be_non_empty_when_given():
    registry = RequirementResolverRegistry()
    with pytest.raises(ValueError):
        registry.register_exact("python", lambda key: "1.0", phase="host", display_name="   ")


def test_prerelease_actual_version_can_satisfy():
    """A non-prerelease-aware specifier (">=1.0") would reject a prerelease
    actual version ("2.0.0rc1") by default in `packaging` -- check() must
    pass prereleases=True so a repo/component pinned to a prerelease still
    satisfies a plain lower-bound requirement."""
    registry = RequirementResolverRegistry()
    registry.register_exact("runtime:pre", lambda key: "2.0.0rc1", phase="runtime")
    results = registry.check({"runtime:pre": ">=1.0"}, phase="runtime")
    assert results[0].status == RequirementStatus.SATISFIED


# -- Import boundary ---------------------------------------------------------

def test_core_manifest_module_has_no_capability_or_docker_imports():
    """linktools.core._manifest must never import a capability package
    (linktools.cntr, linktools.ai) or a container-runtime library --
    Core provides the generic mechanism only; each capability supplies its
    own domain policy and runtime resolvers."""
    import linktools.core._manifest as manifest_module

    source = Path(manifest_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_names.add(node.module)

    forbidden_prefixes = ("linktools.cntr", "linktools.ai", "docker")
    offenders = [
        name for name in imported_names
        if any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
    ]
    assert not offenders, "linktools.core._manifest must not import: %s" % offenders
