"""Plan serialization and read-only lifecycle rendering helpers."""
import dataclasses
import json

from ..container import ContainerError
from ..execution.model import ExecutionPlan


def _plan_to_dict(plan: "ExecutionPlan") -> dict:
    data = dataclasses.asdict(plan)
    for command in data.get("commands", []):
        command.pop("args", None)
    return data


def render_plan(logger, plan: "ExecutionPlan") -> None:
    logger.info("action: %s" % plan.action)
    logger.info("project: %s" % plan.project)
    logger.info("full: %s" % plan.full)
    if plan.targets:
        logger.info("targets: %s" % ", ".join(plan.targets))
    logger.info("resolved containers: %s" % ", ".join(plan.resolved_containers))
    if plan.services:
        logger.info("services: %s" % ", ".join(plan.services))
    if plan.compose_files:
        logger.info("compose files:")
        for path in plan.compose_files:
            logger.info("  %s" % path)
    if plan.artifacts:
        logger.info("artifacts:")
        for artifact in plan.artifacts:
            logger.info("  [%s] %s (%s, %s)" % (artifact.change, artifact.path, artifact.kind, artifact.container))
    if plan.commands:
        logger.info("commands:")
        for command in plan.commands:
            logger.info("  [%s] %s" % (command.phase, " ".join(command.display_args)))
    if plan.hooks:
        logger.info("hooks:")
        for hook in plan.hooks:
            logger.info("  [%s] %s: %s%s" % (hook.phase, hook.container or "(manager)", hook.name,
                                                " (opaque)" if hook.opaque else ""))
    logger.info("preflight: %s" % plan.preflight)
    for warning in plan.warnings:
        logger.info("[WARN] %s" % warning)


def maybe_dry_run(manager, logger, action, names=None, build=True, pull=False, dry_run=False, as_json=False):
    if as_json and not dry_run:
        raise ContainerError("--json requires --dry-run")
    if not dry_run:
        return False
    plan = manager.planner.plan(action, names=names, build=build, pull=pull)
    if as_json:
        print(json.dumps(_plan_to_dict(plan), indent=2, sort_keys=True))
    else:
        render_plan(logger, plan)
    if plan.preflight == "failed":
        raise ContainerError("Compose preflight failed")
    return True
