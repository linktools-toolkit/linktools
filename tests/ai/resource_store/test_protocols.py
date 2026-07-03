from linktools.ai.resource_store.protocols import (
    DeleteOp,
    MoveOp,
    Operation,
    PutOp,
    ResourceBackend,
    ResourceFile,
)


def test_resource_file_is_frozen_with_three_fields():
    rf = ResourceFile(path="/skill/my-skill/SKILL.md", content="hello", version=1)
    assert rf.path == "/skill/my-skill/SKILL.md"
    assert rf.content == "hello"
    assert rf.version == 1
    try:
        rf.version = 2
        assert False, "ResourceFile must be frozen"
    except AttributeError:
        pass


def test_operation_types_construct():
    put = PutOp(path="/skill/a/SKILL.md", content="x")
    delete = DeleteOp(path="/skill/a/SKILL.md")
    move = MoveOp(src_path="/skill/a/SKILL.md", dst_path="/skill/b/SKILL.md")
    assert isinstance(put, PutOp)
    assert isinstance(delete, DeleteOp)
    assert isinstance(move, MoveOp)
    assert Operation is not None


def test_resource_backend_is_runtime_checkable_protocol():
    class _Impl:
        async def propfind(self, path): return []
        async def get(self, path): return None
        async def get_at_version(self, path, version): return None
        async def get_by_name(self, namespace, name): return []
        async def put(self, path, content, *, updated_by="engine"): return None
        async def delete(self, path, *, updated_by="engine"): return False
        async def move(self, src_path, dst_path, *, updated_by="engine"): return None
        async def list_since(self, since): return []
        async def apply_batch(self, ops, *, updated_by="engine"): return []
        async def get_revision(self): return 0

    assert isinstance(_Impl(), ResourceBackend)
