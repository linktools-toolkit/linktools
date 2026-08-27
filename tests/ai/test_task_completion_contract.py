#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task completion identity contract."""

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.task import TaskCompletionLedger


def test_task_completion_uses_owner_fence_and_result_identity() -> None:
    ledger = TaskCompletionLedger()

    first = ledger.complete("task", "owner", 1, "digest")
    assert ledger.complete("task", "owner", 1, "digest") == first

    with pytest.raises(AIError) as error:
        ledger.complete("task", "owner", 1, "other")
    assert error.value.code is ErrorCode.TASK_RESULT_CONFLICT
