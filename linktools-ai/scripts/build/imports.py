"""Static downstream import inventory and migration gate."""

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImportRecord:
    file: str
    module: str
    symbol: str
    alias: "str | None"
    line: int


def scan_imports(root: Path, package: str) -> 'tuple[ImportRecord, ...]':
    records: list[ImportRecord] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in {".venv", "__pycache__", "build", "dist"} for part in path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        relative = path.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (node.module == package or node.module.startswith(f"{package}."))
            ):
                for item in node.names:
                    records.append(ImportRecord(relative, node.module, item.name, item.asname, node.lineno))
            elif isinstance(node, ast.Import):
                for item in node.names:
                    if item.name == package or item.name.startswith(f"{package}."):
                        records.append(ImportRecord(relative, item.name, "", item.asname, node.lineno))
    return tuple(records)


def write_inventory(root: Path, package: str, output: Path) -> 'tuple[ImportRecord, ...]':
    records = scan_imports(root, package)
    output.write_text(json.dumps([asdict(record) for record in records], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = write_inventory(arguments.root, arguments.package, arguments.output)
    print(json.dumps({"files": len({record.file for record in records}), "imports": len(records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ImportRecord", "main", "scan_imports", "write_inventory"]
