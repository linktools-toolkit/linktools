#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optional MySQL integration surface; live execution is environment-gated."""

import os

import pytest

from linktools.ai import RuntimePersistenceConfig, open_runtime_resources


def test_mysql_config_redacts_credentials() -> None:
    config = RuntimePersistenceConfig.mysql("mysql+asyncmy://user:secret@example.test/db", namespace="namespace", deployment_id="deployment")
    assert "secret" not in repr(config)
    assert "secret" not in str(config)


@pytest.mark.asyncio
async def test_mysql_live_not_run_without_environment() -> None:
    url = os.getenv("LINKTOOLS_AI_TEST_MYSQL_URL")
    if not url:
        pytest.skip("not_run_no_environment_accepted")
    async with open_runtime_resources(RuntimePersistenceConfig.mysql(url, namespace="test", deployment_id="pytest")):
        pass
