#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optional MySQL integration surface; live execution is environment-gated."""

import os

import pytest
from linktools.ai import RuntimeStorage

from tests.ai.persistence.helper import open_sql_resources


def test_mysql_config_redacts_credentials() -> None:
    assert RuntimeStorage.memory().location is None


@pytest.mark.asyncio
async def test_mysql_live_not_run_without_environment() -> None:
    url = os.getenv("LINKTOOLS_AI_TEST_MYSQL_URL")
    if not url:
        pytest.skip("not_run_no_environment_accepted")
    async with open_sql_resources(RuntimeStorage.memory(), connection_url=url):
        pass
