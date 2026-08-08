#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optional PostgreSQL integration surface; live execution is environment-gated."""

import os

import pytest

from linktools.ai import RuntimePersistenceConfig
from tests.ai.persistence.helper import open_sql_resources


@pytest.mark.asyncio
async def test_postgresql_live_not_run_without_environment() -> None:
    url = os.getenv("LINKTOOLS_AI_TEST_POSTGRESQL_URL")
    if not url:
        pytest.skip("not_run_no_environment_accepted")
    async with open_sql_resources(RuntimePersistenceConfig.postgresql(namespace="test", deployment_id="pytest"), connection_url=url):
        pass
