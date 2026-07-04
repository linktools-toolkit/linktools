from linktools.ai.resource.protocols import (
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


def test_resource_backend_subclass_implementing_all_abstract_methods_is_instantiable():
    class _Impl(ResourceBackend):
        async def get(self, path, version=None): return None
        async def list(self, *, pattern=None, since=None): return []
        async def put(self, path, content, *, updated_by=""): return None
        async def delete(self, path, *, updated_by=""): return False
        async def move(self, src_path, dst_path, *, updated_by=""): return None
        async def apply_batch(self, ops, *, updated_by=""): return []
        async def revision(self): return 0

    assert isinstance(_Impl(), ResourceBackend)
