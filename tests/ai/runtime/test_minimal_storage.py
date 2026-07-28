from linktools.ai.runtime import LocalDirectoryStorage
import pytest


def test_minimal_local_storage_has_only_execution_and_is_lazy(tmp_path):
    root = tmp_path / ".linktools"
    storage = LocalDirectoryStorage(root)
    assert storage.tools is None
    assert storage.tasks is None
    assert storage.memory is None
    assert storage.artifacts is None
    assert not root.exists()


@pytest.mark.asyncio
async def test_local_storage_initialization_is_explicit(tmp_path):
    storage = LocalDirectoryStorage(tmp_path / ".linktools")
    await storage.initialize_storage()
    assert (tmp_path / ".linktools/execution").is_dir()
