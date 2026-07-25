#!/usr/bin/env python3
"""Run the review-spec alias and non-tool CLI performance gate."""

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


RUNS = 30


def prepare_baseline(root, current):
    """Supply generated metadata/assets when benchmarking an old checkout."""
    source_metadata = current / "linktools/src/linktools/metadata.py"
    target_metadata = root / "linktools/src/linktools/metadata.py"
    if source_metadata.is_file() and not target_metadata.exists():
        target_metadata.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source_metadata), str(target_metadata))
    legacy_assets = root / "linktools/src/linktools/assets/tools.json"
    current_assets = current / "linktools/src/linktools/assets/tools/linktools.json"
    if current_assets.is_file() and not legacy_assets.exists():
        shutil.copyfile(str(current_assets), str(legacy_assets))


def measure(root, python, label, args):
    values = []
    for index in range(RUNS):
        data_path = Path(tempfile.gettempdir()) / ("linktools-perf-%s-%d" % (label, index))
        environment = dict(os.environ, DEBUG="false", SHELL="/bin/bash",
                           LINKTOOLS_DATA_PATH=str(data_path),
                           PYTHONPATH=str(root / "linktools/src"))
        started = time.perf_counter()
        result = subprocess.run([python, "-m", "linktools.cli.env", *args], cwd=str(root),
                                env=environment, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        values.append((time.perf_counter() - started) * 1000)
        if result.returncode:
            raise SystemExit("%s failed on run %d with status %d" % (label, index, result.returncode))
    return {"median_ms": statistics.median(values), "p95_ms": sorted(values)[28]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, default=Path.cwd())
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prepare_baseline(args.baseline_root, args.current_root)
    python = sys.executable
    result = {"baseline": {}, "current": {}}
    for label, root in (("baseline", args.baseline_root), ("current", args.current_root)):
        result[label]["alias"] = measure(root, python, label + "-alias", ["alias", "--reload"])
        result[label]["non_tool_help"] = measure(root, python, label + "-help", ["--help"])
    for command in ("alias", "non_tool_help"):
        baseline = result["baseline"][command]["median_ms"]
        current = result["current"][command]["median_ms"]
        if current - baseline > max(5.0, baseline * 0.05):
            raise SystemExit("performance gate failed for %s: %.2f -> %.2f ms" %
                             (command, baseline, current))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
