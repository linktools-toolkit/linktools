#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr lock [--check] [--json]`` / ``ct-cntr diff [--json]`` (Spec Part
VI). Lock is fully opt-in (section 43): up/restart/down never require it,
`repo update` never modifies it, and a missing lock is never a warning by
itself -- only `lock --check`/`diff` treat a missing lock as something to
report.
"""
import dataclasses
import json
from typing import TYPE_CHECKING

from linktools.cli import BaseCommand, CommandParser
from ..container import ContainerError
from ..lock.diff import LockDiff, compute_diff
from . import _shared

if TYPE_CHECKING:
    from argparse import Namespace


def _diff_to_dict(diff: "LockDiff") -> dict:
    return dataclasses.asdict(diff)


def render_diff(logger, diff: "LockDiff", persisted_missing: bool = False) -> None:
    if persisted_missing:
        logger.info("No persisted lock found; nothing to compare against.")
    if diff.cntr_version_changed:
        logger.info(f"linktools-cntr: {diff.old_cntr_version} -> {diff.new_cntr_version}")
    for drift in diff.repository_drifts:
        logger.info(f"repo `{drift.url}` {drift.field}: {drift.old} -> {drift.new}")
    for url in diff.repositories_added:
        logger.info(f"repo added: {url}")
    for url in diff.repositories_removed:
        logger.info(f"repo removed: {url}")
    for name in diff.containers_added:
        logger.info(f"container added: {name}")
    for name in diff.containers_removed:
        logger.info(f"container removed: {name}")
    for artifact in diff.artifact_drifts:
        logger.info(f"artifact {artifact.change}: {artifact.path}")
    if diff.is_empty and not persisted_missing:
        logger.info("No drift detected.")


class LockCommand(BaseCommand):
    """
    generate or check the deployment lock (container.lock.json)
    """

    @property
    def name(self) -> str:
        return "lock"

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--check", action="store_true", default=False,
                            help="compare current state against the persisted lock; do not write it")
        parser.add_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")

    def run(self, args: "Namespace"):
        manager = _shared.manager

        if args.check:
            persisted = manager.lock_store.load()
            current = manager.lock_store.build()
            diff = compute_diff(persisted, current)
            if args.as_json:
                print(json.dumps(_diff_to_dict(diff), indent=2, sort_keys=True))
            else:
                render_diff(self.logger, diff, persisted_missing=(persisted is None))
            if persisted is None:
                raise ContainerError("No persisted lock found")
            if not diff.is_empty:
                raise ContainerError("Lock drift detected")
            return

        lock, preflight = manager.lock_store.build_and_preflight()
        if preflight == "failed":
            raise ContainerError("Compose preflight failed; lock not written")

        manager.lock_store.write(lock)
        if args.as_json:
            print(json.dumps(manager.lock_store.to_dict(lock), indent=2, sort_keys=True))
        else:
            self.logger.info(f"Lock written to {manager.lock_store.path}")
            self.logger.info(f"project: {lock.project}")
            self.logger.info(f"linktools-cntr: {lock.linktools_cntr}")
            self.logger.info(f"repositories: {len(lock.repositories)}")
            self.logger.info(f"containers: {len(lock.containers)}")
            self.logger.info(f"artifacts: {len(lock.artifacts)}")


class DiffCommand(BaseCommand):
    """
    show drift between the current state and the persisted lock (read-only)
    """

    @property
    def name(self) -> str:
        return "diff"

    def init_arguments(self, parser: "CommandParser") -> None:
        parser.add_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")

    def run(self, args: "Namespace"):
        manager = _shared.manager
        persisted = manager.lock_store.load()
        current = manager.lock_store.build()
        diff = compute_diff(persisted, current)
        if args.as_json:
            print(json.dumps(_diff_to_dict(diff), indent=2, sort_keys=True))
        else:
            render_diff(self.logger, diff, persisted_missing=(persisted is None))
