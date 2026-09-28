#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The removed approval context reader must not return as a second owner."""

from linktools.ai.runtime._local import LocalExecutionBackend


def test_local_backend_does_not_expose_approval_context_reader() -> None:
    assert not hasattr(LocalExecutionBackend, "tool_approvals")
