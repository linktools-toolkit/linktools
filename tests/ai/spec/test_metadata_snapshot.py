import pytest

from linktools.ai.spec.document import SpecDocumentInfo, SpecDocumentChange
from linktools.ai.spec.revision import MetadataSnapshot


class Repository:
    def __init__(self):
        self.revision = 1
        self.values = {"a": SpecDocumentInfo("a", "agent", 1, "e1")}
        self.full_reads = 0
        self.delta_reads = 0

    async def current_revision(self):
        return self.revision

    async def list_info(self, *, kind=None):
        self.full_reads += 1
        return tuple(self.values.values())

    async def list_changes(self, *, after_revision, through_revision):
        self.delta_reads += 1
        return (SpecDocumentChange(2, "a", self.values["a"]),)


@pytest.mark.asyncio
async def test_metadata_stat_and_list_share_revision_snapshot():
    repo = Repository()
    snapshot = MetadataSnapshot(repo, revision=repo, changes=repo)
    assert await snapshot.get("a") is not None
    assert repo.full_reads == 1
    repo.revision = 2
    assert await snapshot.get("a") is not None
    assert repo.delta_reads == 1
