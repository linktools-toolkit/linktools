from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "linktools-ai/src/linktools/ai"


def test_domain_sqlalchemy_tables_use_storage_prefix():
    files = (
        ROOT / "execution/persistence/sqlalchemy.py",
        ROOT / "agent/tool/persistence/sqlalchemy.py",
        ROOT / "tasks/persistence/sqlalchemy.py",
        ROOT / "agent/memory/persistence/sqlalchemy.py",
        ROOT / "spec/persistence/sqlalchemy.py",
    )
    for path in files:
        source = path.read_text(encoding="utf-8")
        assert "TABLE_PREFIX" in source
        assert '__tablename__ = "' not in source
