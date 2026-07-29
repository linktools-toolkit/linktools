from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "linktools-ai/src/linktools/ai"


def test_new_storage_protocols_do_not_import_backends():
    for relative in (
        "artifact/store.py",
        "spec/store.py",
        "execution/store.py",
        "agent/memory/store.py",
        "tasks/store.py",
        "tool/store.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "persistence.local" not in text
        assert "persistence.sqlalchemy" not in text
        assert "from linktools.ai" not in text


def test_new_agent_spec_entrypoint_is_the_only_public_markdown_wrapper():
    codec = (ROOT / "agent/codec.py").read_text(encoding="utf-8")
    assert "def parse_agent_spec_markdown" in codec
    assert "_split_frontmatter" not in codec


def test_storage_kernel_contains_no_domain_models_or_backend_rows():
    storage = ROOT / "storage"
    for path in storage.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "class MemoryRow" not in text
        assert "class StorageObject" not in text
        assert "class RunRow" not in text
        assert "from ..memory" not in text
        assert "from ..capability" not in text
        assert "from ..execution" not in text


def test_each_domain_exposes_one_composed_store_and_storage_contracts():
    expected = {
        "spec/store.py": ("SpecReader", "SpecWriter", "SpecStore"),
        "tool/store.py": ("ToolStateBackend", "ToolStateStore"),
        "tasks/store.py": ("TaskBackend", "TaskStore"),
        "agent/memory/store.py": ("MemoryBackend", "MemoryStore"),
        "artifact/store.py": ("ArtifactBackend", "ArtifactStore"),
        "execution/store.py": ("ExecutionBackend", "ExecutionStore"),
    }
    for relative, names in expected.items():
        text = (ROOT / relative).read_text(encoding="utf-8")
        for name in names:
            assert f"class {name}" in text
        assert "StorageComposition" in text
        assert "__getattr__" not in text


def test_sqlalchemy_adapters_share_one_declarative_base():
    base = (ROOT / "storage/sqlalchemy/base.py").read_text(encoding="utf-8")
    assert "class Base(DeclarativeBase)" in base
    assert "id: Mapped[int]" in base
    assert "created_at: Mapped[datetime]" in base
    assert "updated_at: Mapped[datetime]" in base
    for relative in (
        "spec/persistence/sqlalchemy.py",
        "execution/persistence/sqlalchemy.py",
        "agent/memory/persistence/sqlalchemy.py",
        "tasks/persistence/sqlalchemy.py",
        "tool/persistence/sqlalchemy.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "storage.sqlalchemy.base import Base" in text
        assert "class Base(DeclarativeBase)" not in text
