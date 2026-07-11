#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit, testable CLI command ordering.

String orders (not integers) so a command framework that mixes an explicit
order with its own default string order never hits a type-comparison error.
"""

ROOT_COMMAND_ORDER = {
    "list": "010-list",
    "status": "020-status",
    "add": "030-add",
    "remove": "040-remove",
    "up": "050-up",
    "restart": "060-restart",
    "down": "070-down",
    "exec": "080-exec",
    "compose": "090-compose",
    "plan": "100-plan",
    "config": "110-config",
    "repo": "120-repo",
    "doctor": "130-doctor",
}

CONFIG_COMMAND_ORDER = {
    "list": "010-list",
    "get": "020-get",
    "explain": "030-explain",
    "set": "040-set",
    "unset": "050-unset",
    "edit": "060-edit",
    "reload": "070-reload",
    "validate": "080-validate",
}

REPO_COMMAND_ORDER = {
    "list": "010-list",
    "status": "020-status",
    "add": "030-add",
    "update": "040-update",
    "remove": "050-remove",
    "validate": "060-validate",
}

assert len(set(ROOT_COMMAND_ORDER.values())) == len(ROOT_COMMAND_ORDER)
assert len(set(CONFIG_COMMAND_ORDER.values())) == len(CONFIG_COMMAND_ORDER)
assert len(set(REPO_COMMAND_ORDER.values())) == len(REPO_COMMAND_ORDER)
