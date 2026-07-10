# -*- coding: utf-8 -*-
"""Tests for ConfigMigration (v2 §3.3, PR-2 §4)."""
import configparser
import json
import os

import pytest

from linktools.core import ConfigStore
from linktools.core._locks import LockManager
from linktools.core import ConfigMigration


@pytest.fixture
def setup(tmp_path):
    store = ConfigStore(tmp_path / "new.json", lock_manager=LockManager(tmp_path / "l"))
    old = tmp_path / "old.cfg"
    return store, old, tmp_path


def _write_sections(path, sections):
    """sections: {section_name: {key: value}}."""
    parser = configparser.ConfigParser()
    parser.optionxform = str  # preserve case
    for name, entries in sections.items():
        parser[name] = entries
    with open(path, "w") as f:
        parser.write(f)


def _write_old(path, entries, section="CONTAINER.CACHE"):
    _write_sections(path, {section: entries})


# --------------------------------------------------------------------------- #
# inspect (§4.3: section-qualified keys)
# --------------------------------------------------------------------------- #

def test_inspect_finds_section_qualified_keys(setup):
    store, old, _ = setup
    _write_old(old, {"HOST": "1.2.3.4", "PORT": "8080"})
    info = ConfigMigration(store).inspect(old)
    assert info["file_exists"] is True
    assert info["count"] == 2
    assert "CONTAINER.CACHE.HOST" in info["keys"]
    assert "CONTAINER.CACHE.PORT" in info["keys"]


def test_inspect_missing_file(setup):
    store, old, _ = setup
    info = ConfigMigration(store).inspect(old)
    assert info["file_exists"] is False
    assert info["count"] == 0


def test_same_key_in_different_sections_does_not_collide(setup):
    store, old, _ = setup
    _write_sections(old, {
        "MAIN.CACHE": {"PORT": "1"},
        "CONTAINER.CACHE": {"PORT": "2"},
    })
    report = ConfigMigration(store).migrate(old)
    # both migrated distinctly to their own namespace, not collapsed onto one key
    assert store.get("main.PORT") == "1"
    assert store.get("container.PORT") == "2"
    assert len([e for e in report["entries"] if e["reason"] == "mapped_generic"]) == 2


# --------------------------------------------------------------------------- #
# migrate (§4.5: map / preserve / skip)
# --------------------------------------------------------------------------- #

def test_migrate_writes_to_config_store(setup):
    store, old, _ = setup
    _write_old(old, {"HOST": "1.2.3.4"})
    report = ConfigMigration(store).migrate(old, key_map={"CONTAINER.CACHE.HOST": "network.host"})
    assert "CONTAINER.CACHE.HOST" in report["migrated"]
    assert store.get("network.host") == "1.2.3.4"


def test_migrate_unmapped_cache_key_maps_generically(setup):
    # Not one of ConfigMigration's built-in exceptions and not in any
    # caller-supplied key_map -- e.g. a custom homelab container's own field --
    # still migrates to a real, readable "<namespace>.<KEY>" location rather
    # than being dropped into a dead legacy.* key.
    store, old, _ = setup
    _write_old(old, {"UNKNOWN_KEY": "val"})
    report = ConfigMigration(store).migrate(old)
    assert "CONTAINER.CACHE.UNKNOWN_KEY" in report["migrated"]
    assert store.get("container.UNKNOWN_KEY") == "val"
    entry = [e for e in report["entries"] if e["old_key"] == "CONTAINER.CACHE.UNKNOWN_KEY"][0]
    assert entry["reason"] == "mapped_generic"


def test_migrate_unrecognized_section_shape_falls_back_to_legacy(setup):
    # A section that doesn't follow the "*.CACHE" convention at all has no
    # namespace to derive -- this is the one case still preserved dead.
    store, old, _ = setup
    _write_sections(old, {"SOME.OTHER.SECTION": {"KEY": "val"}})
    report = ConfigMigration(store).migrate(old)
    assert store.get("legacy.some.other.section.key") == "val"
    entry = report["entries"][0]
    assert entry["reason"] == "unknown_key_preserved"


def test_migrate_skips_existing(setup):
    store, old, _ = setup
    store.set("network.host", "newer")
    _write_old(old, {"HOST": "older"})
    report = ConfigMigration(store).migrate(old, key_map={"CONTAINER.CACHE.HOST": "network.host"})
    assert "CONTAINER.CACHE.HOST" in report["skipped"]
    assert store.get("network.host") == "newer"  # not overwritten


def test_migrate_flags_secret_keys(setup):
    store, old, _ = setup
    _write_old(old, {"DB_PASSWORD": "p4ss", "HOST": "h"})
    report = ConfigMigration(store).migrate(old)
    by_old = {e["old_key"]: e for e in report["entries"]}
    assert by_old["CONTAINER.CACHE.DB_PASSWORD"]["secret"] is True
    assert by_old["CONTAINER.CACHE.HOST"]["secret"] is False


def test_migrate_dry_run_writes_nothing(setup):
    store, old, _ = setup
    _write_old(old, {"HOST": "1.2.3.4"})
    report = ConfigMigration(store).migrate(old, key_map={"CONTAINER.CACHE.HOST": "network.host"}, dry_run=True)
    assert "CONTAINER.CACHE.HOST" in report["migrated"]
    assert "network.host" not in store  # nothing written


# --------------------------------------------------------------------------- #
# fix-plan PR-1 §1.3.2/§1.3.3: full-key priority + ambiguous bare-key safety
# --------------------------------------------------------------------------- #

def test_full_key_map_beats_bare_key_map(setup):
    store, old, _ = setup
    _write_old(old, {"HOST": "h"})
    # both full and bare provided for the same source key; the full key wins
    ConfigMigration(store).migrate(old, key_map={
        "CONTAINER.CACHE.HOST": "full.wins",
        "HOST": "bare.loses",
    })
    assert store.get("full.wins") == "h"
    assert "bare.loses" not in store


def test_ambiguous_bare_key_not_auto_mapped(setup):
    store, old, _ = setup
    _write_sections(old, {
        "MAIN.CACHE": {"CUSTOM": "a"},
        "CONTAINER.CACHE": {"CUSTOM": "b"},
    })
    # bare CUSTOM is ambiguous (2 sections) and only a bare map is given -> the
    # bare fallback is refused; each still migrates to its own namespace via
    # the generic per-section rule, so the two sections never collapse onto
    # the one bare key the caller supplied.
    ConfigMigration(store).migrate(old, key_map={"CUSTOM": "should.not.win"})
    assert "should.not.win" not in store
    assert store.get("main.CUSTOM") == "a"
    assert store.get("container.CUSTOM") == "b"


def test_ambiguous_bare_key_mapped_via_full_keys(setup):
    store, old, _ = setup
    _write_sections(old, {
        "MAIN.CACHE": {"HOST": "a"},
        "CONTAINER.CACHE": {"HOST": "b"},
    })
    # explicit full-key mappings disambiguate the same bare key
    ConfigMigration(store).migrate(old, key_map={
        "MAIN.CACHE.HOST": "core.host",
        "CONTAINER.CACHE.HOST": "container.host",
    })
    assert store.get("core.host") == "a"
    assert store.get("container.host") == "b"


def test_migrate_uses_one_batch_save(setup):
    # fix-plan §1.3.4: plan once, write once via store.save (not per-key set)
    store, old, _ = setup
    calls = {"save": 0, "set": 0}
    real_save, real_set = store.save, store.set

    def spy_save(**kw):
        calls["save"] += 1
        return real_save(**kw)

    def spy_set(key, value):
        calls["set"] += 1
        return real_set(key, value)

    store.save = spy_save
    store.set = spy_set
    _write_old(old, {"HOST": "1", "PORT": "2"})
    ConfigMigration(store).migrate(old, key_map={
        "CONTAINER.CACHE.HOST": "h", "CONTAINER.CACHE.PORT": "p"})
    assert calls["save"] == 1  # one atomic batch write
    assert calls["set"] == 0   # no per-key writes


def test_secret_entry_has_no_raw_value_in_report(setup):
    # fix-plan §1.5: secret raw values must never appear in the report
    store, old, _ = setup
    _write_old(old, {"DB_PASSWORD": "p4ss"})
    report = ConfigMigration(store).migrate(old)
    entry = [e for e in report["entries"] if e["secret"]][0]
    assert "value" not in entry


def test_main_cache_host_not_mapped_to_container_host_by_default(setup):
    # A section-sensitive bare key (HOST) is NOT bare-mapped by default, so a
    # stray MAIN.CACHE.HOST must NOT be pulled onto container.HOST -- the
    # section-qualified generic rule sends it to its own main.HOST instead.
    store, old, _ = setup
    _write_sections(old, {"MAIN.CACHE": {"HOST": "main-host"}})
    ConfigMigration(store).migrate(old)
    assert "container.HOST" not in store
    assert store.get("main.HOST") == "main-host"


def test_container_host_mapped_only_via_full_key(setup):
    # The container HOST still migrates correctly when addressed by its full key.
    store, old, _ = setup
    _write_sections(old, {"CONTAINER.CACHE": {"HOST": "c-host"}})
    ConfigMigration(store).migrate(old)
    assert store.get("container.HOST") == "c-host"


# --------------------------------------------------------------------------- #
# backup (§4.6: never overwrite) + rollback
# --------------------------------------------------------------------------- #

def test_backup_writes_to_unique_migration_dir(setup):
    store, old, tmp = setup
    _write_old(old, {"X": "1"})
    mig = ConfigMigration(store, config_dir=tmp)
    bak = mig.backup(old)
    # lands under <config_dir>/migrations/<id>/old-config.backup
    assert "migrations" in bak
    assert bak.endswith("old-config.backup")
    assert os.path.isfile(bak)
    # report.json sidecar exists with a sha256 + migration_id
    report_json = json.loads(open(os.path.join(os.path.dirname(bak), "report.json")).read())
    assert "sha256" in report_json and "migration_id" in report_json


def test_backup_does_not_overwrite_previous(setup):
    store, old, tmp = setup
    _write_old(old, {"X": "1"})
    mig = ConfigMigration(store, config_dir=tmp)
    first = mig.backup(old)
    second = mig.backup(old)  # different migration_id -> different dir
    assert first != second
    assert os.path.isfile(first) and os.path.isfile(second)


def test_backup_and_rollback(setup):
    store, old, tmp = setup
    _write_old(old, {"X": "1"})
    mig = ConfigMigration(store, config_dir=tmp)
    bak = mig.backup(old)
    os.remove(old)
    assert not old.exists()
    mig.rollback(bak, old)
    assert old.exists()


# --------------------------------------------------------------------------- #
# verify (§4.8: full check, not a spot-check)
# --------------------------------------------------------------------------- #

def test_verify_store_readable(setup):
    store, old, _ = setup
    store.set("a", "1")
    store.set("b", "2")
    assert ConfigMigration(store).verify() is True


def test_verify_report_checks_every_claimed_key(setup):
    store, old, _ = setup
    _write_old(old, {"HOST": "h", "PORT": "8080"})
    mig = ConfigMigration(store)
    report = mig.migrate(old, key_map={"HOST": "network.host", "PORT": "network.port"})
    assert mig.verify(report) is True
    # if a claimed key is missing, verify must fail
    store.remove("network.port")
    assert mig.verify(report) is False


# --------------------------------------------------------------------------- #
# Migrated targets must actually be readable back through PersistentSource
# (regression: a lowercase/dotted target key like "container.docker_type" is
# written but PersistentSource only ever looks up "<namespace>.<FIELD_NAME>"
# verbatim, so the migrated value could never be read back).
# --------------------------------------------------------------------------- #

def test_migrated_generic_targets_are_readable_via_persistent_source(setup):
    from linktools.core._config import PersistentSource

    store, old, _ = setup
    _write_sections(old, {
        "MAIN.CACHE": {"DEBUG": "true"},
        "CONTAINER.CACHE": {"DOCKER_TYPE": "podman", "HOST": "1.2.3.4"},
    })
    ConfigMigration(store).migrate(old)

    main_source = PersistentSource(store, "main")
    assert main_source.get("DEBUG") == ("true", True)

    container_source = PersistentSource(store, "container")
    assert container_source.get("DOCKER_TYPE") == ("podman", True)
    assert container_source.get("HOST") == ("1.2.3.4", True)


def test_installed_state_keys_migrate_bare(setup):
    # ContainerManager reads these via _load_setting, straight off the store
    # with no namespace prefix (see _migrate.py / manager._persistent_store).
    store, old, _ = setup
    _write_sections(old, {"CONTAINER.CACHE": {"INSTALLED_CONTAINERS": '["nginx"]'}})
    ConfigMigration(store).migrate(old)
    assert store.get("INSTALLED_CONTAINERS") == '["nginx"]'


def test_flare_doamin_misspelling_renamed_to_flare_domain(setup):
    # A built-in exception to the generic rule: the legacy misspelling must be
    # renamed on the way in, not migrated verbatim to container.FLARE_DOAMIN
    # (a key nothing reads).
    store, old, _ = setup
    _write_sections(old, {"CONTAINER.CACHE": {"FLARE_DOAMIN": "example.com"}})
    report = ConfigMigration(store).migrate(old)
    assert store.get("container.FLARE_DOMAIN") == "example.com"
    assert "container.FLARE_DOAMIN" not in store
    entry = [e for e in report["entries"] if e["old_key"] == "CONTAINER.CACHE.FLARE_DOAMIN"][0]
    assert entry["new_key"] == "container.FLARE_DOMAIN"
    assert entry["reason"] == "mapped"
