#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``ct-cntr plan up/restart/down``: shows what an action would
do without doing it. Never executes a Docker write operation, a lifecycle
hook, or writes a generated artifact/state file."""
import dataclasses
import json
from typing import TYPE_CHECKING

from linktools.cli import BaseCommandGroup, subcommand, subcommand_argument
from linktools.cli.argparse import BooleanOptionalAction, LazyChoices
from ..container import ContainerError
from ..execution.model import ExecutionPlan
from . import _shared

if TYPE_CHECKING:
    pass


def _plan_to_dict(plan: "ExecutionPlan") -> dict:
    # Only display_args is ever surfaced -- args exists on PlannedCommand
    # purely for exact-argv comparison in tests, and must never appear in
    # a rendered/serialized plan even though it's already redacted.
    data = dataclasses.asdict(plan)
    for command in data.get("commands", []):
        command.pop("args", None)
    return data


def render_plan(logger, plan: "ExecutionPlan") -> None:
    logger.info(f"action: {plan.action}")
    logger.info(f"project: {plan.project}")
    logger.info(f"full: {plan.full}")
    if plan.targets:
        logger.info(f"targets: {', '.join(plan.targets)}")
    logger.info(f"resolved containers: {', '.join(plan.resolved_containers)}")
    if plan.services:
        logger.info(f"services: {', '.join(plan.services)}")
    if plan.compose_files:
        logger.info("compose files:")
        for path in plan.compose_files:
            logger.info(f"  {path}")

    if plan.artifacts:
        logger.info("artifacts:")
        for artifact in plan.artifacts:
            logger.info(f"  [{artifact.change}] {artifact.path} ({artifact.kind}, {artifact.container})")

    if plan.commands:
        logger.info("commands:")
        for command in plan.commands:
            logger.info(f"  [{command.phase}] {' '.join(command.display_args)}")

    if plan.hooks:
        logger.info("hooks:")
        for hook in plan.hooks:
            scope = hook.container or "(manager)"
            logger.info(f"  [{hook.phase}] {scope}: {hook.name}{' (opaque)' if hook.opaque else ''}")

    logger.info(f"preflight: {plan.preflight}")
    for warning in plan.warnings:
        logger.info(f"[WARN] {warning}")


def maybe_dry_run(manager, logger, action: str, names=None, build: bool = True, pull: bool = False,
                  dry_run: bool = False) -> bool:
    """If ``dry_run``, render the plan (same Planner/model as ``ct-cntr
    plan``) and return True so the caller stops instead of executing the
    real action; otherwise return False."""
    if not dry_run:
        return False
    plan = manager.planner.plan(action, names=names, build=build, pull=pull)
    render_plan(logger, plan)
    if plan.preflight == "failed":
        raise ContainerError("Compose preflight failed")
    return True


class PlanCommand(BaseCommandGroup):
    """
    show what up/restart/down would do, without doing it (read-only)
    """

    @property
    def name(self) -> str:
        return "plan"

    @subcommand("up", help="plan deploying installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_up(self, names: "list[str]" = None, build: bool = True, pull: bool = False,
                      as_json: bool = False):
        return self._plan("up", names=names, build=build, pull=pull, as_json=as_json)

    @subcommand("restart", help="plan restarting installed containers")
    @subcommand_argument("--build", action=BooleanOptionalAction, help="build images before starting")
    @subcommand_argument("--pull", action=BooleanOptionalAction,
                         help="always attempt to pull a newer version of the image")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_restart(self, names: "list[str]" = None, build: bool = True, pull: bool = False,
                           as_json: bool = False):
        return self._plan("restart", names=names, build=build, pull=pull, as_json=as_json)

    @subcommand("down", help="plan stopping installed containers")
    @subcommand_argument("--json", dest="as_json", action="store_true", default=False, help="output JSON")
    @subcommand_argument("names", metavar="CONTAINER", nargs="*", help="container name",
                         choices=LazyChoices(_shared.iter_installed_container_names))
    def on_command_down(self, names: "list[str]" = None, as_json: bool = False):
        return self._plan("down", names=names, as_json=as_json)

    def _plan(self, action: str, names=None, build: bool = True, pull: bool = False, as_json: bool = False):
        plan = _shared.manager.planner.plan(action, names=names, build=build, pull=pull)
        if as_json:
            print(json.dumps(_plan_to_dict(plan), indent=2, sort_keys=True))
        else:
            render_plan(self.logger, plan)
        if plan.preflight == "failed":
            raise ContainerError("Compose preflight failed")
