#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify LinkTools release artifacts in the repository dist directory."""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path

import pkginfo
import tomlkit

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIST = _REPO_ROOT / "dist"
_CAPABILITY_RESOURCES = {
    "linktools": "linktools/assets/tools/linktools.json",
    "linktools-common": "linktools/assets/tools/linktools-common.json",
    "linktools-mobile": "linktools/assets/tools/linktools-mobile.json",
}


class _Artifact:
    def __init__(self, path, kind, name, version, requires_python):
        self.path = path
        self.kind = kind
        self.name = name
        self.normalized_name = _normalize_name(name)
        self.version = version
        self.requires_python = requires_python


def _normalize_name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def _project_directories():
    result = {}
    for path in sorted(_REPO_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if path.name != "linktools" and not path.name.startswith("linktools-"):
            continue
        pyproject = path / "pyproject.toml"
        if not pyproject.is_file():
            continue
        data = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
        name = data.get("project", {}).get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("missing project name: %s" % pyproject)
        normalized = _normalize_name(name)
        if normalized in result:
            raise ValueError("duplicate project name: %s" % name)
        result[normalized] = path
    return result


def _artifact_kind(path):
    if path.name.endswith(".whl"):
        return "wheel"
    if path.name.endswith(".tar.gz"):
        return "sdist"
    return None


def _read_artifact(path):
    kind = _artifact_kind(path)
    if kind == "wheel":
        metadata = pkginfo.Wheel(str(path))
    elif kind == "sdist":
        metadata = pkginfo.SDist(str(path))
    else:
        raise ValueError("unsupported artifact: %s" % path)
    name = getattr(metadata, "name", None)
    version = getattr(metadata, "version", None)
    requires_python = getattr(metadata, "requires_python", None)
    if not isinstance(name, str) or not name:
        raise ValueError("artifact has no Name metadata: %s" % path)
    if not isinstance(version, str) or not version:
        raise ValueError("artifact has no Version metadata: %s" % path)
    if requires_python is None:
        requires_python = ""
    if not isinstance(requires_python, str):
        requires_python = str(requires_python)
    return _Artifact(path, kind, name, version, requires_python)


def _load_artifacts(known_projects):
    if not _DIST.is_dir():
        raise ValueError("dist directory does not exist: %s" % _DIST)
    artifacts = []
    for path in sorted(_DIST.iterdir()):
        if path.is_dir():
            continue
        if _artifact_kind(path) is None:
            raise ValueError("unknown file in dist: %s" % path.name)
        artifact = _read_artifact(path)
        if artifact.normalized_name not in known_projects:
            raise ValueError("artifact does not belong to a registered project: %s" % path.name)
        artifacts.append(artifact)
    return artifacts


def _source_metadata(project_path):
    pyproject = project_path / "pyproject.toml"
    data = tomlkit.parse(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    name = project.get("name")
    requires_python = project.get("requires-python")
    if not isinstance(name, str) or not name:
        raise ValueError("missing project name: %s" % pyproject)
    if not isinstance(requires_python, str) or not requires_python:
        raise ValueError("missing requires-python: %s" % pyproject)
    return name, requires_python


def _selected_pairs(artifacts, selected, project_paths):
    pairs = {}
    for project in selected:
        project_artifacts = [artifact for artifact in artifacts if artifact.normalized_name == project]
        wheels = [artifact for artifact in project_artifacts if artifact.kind == "wheel"]
        sdists = [artifact for artifact in project_artifacts if artifact.kind == "sdist"]
        if len(wheels) != 1 or len(sdists) != 1:
            raise ValueError(
                "%s must have exactly one wheel and one sdist, got %d wheel(s) and %d sdist(s)"
                % (project, len(wheels), len(sdists))
            )
        wheel = wheels[0]
        sdist = sdists[0]
        source_name, source_requires_python = _source_metadata(project_paths[project])
        if wheel.name != source_name or sdist.name != source_name:
            raise ValueError("artifact Name mismatch for %s" % project)
        if wheel.version != sdist.version:
            raise ValueError("wheel/sdist Version mismatch for %s" % project)
        if wheel.requires_python != source_requires_python or sdist.requires_python != source_requires_python:
            raise ValueError("Requires-Python mismatch for %s" % project)
        pairs[project] = (wheel, sdist)
    return pairs


def _validate_full_artifact_set(artifacts, known_projects, selected):
    if set(selected) != set(known_projects):
        return
    if len(artifacts) != len(known_projects) * 2:
        raise ValueError("full verification requires exactly two artifacts per registered project")


def _validate_version(pairs, project_paths):
    version = os.environ.get("VERSION")
    if version is None:
        return
    expected = version[1:] if version.startswith("v") else version
    for project, pair in pairs.items():
        wheel, sdist = pair
        if wheel.version != expected or sdist.version != expected:
            raise ValueError("VERSION metadata mismatch for %s: expected %s" % (project, expected))
        version_file = project_paths[project] / ".version"
        if not version_file.is_file():
            raise ValueError("missing version file: %s" % version_file)
        if version_file.read_text(encoding="utf-8") != version:
            raise ValueError(".version mismatch for %s" % project)


def _wheel_resource(wheel, resource):
    with zipfile.ZipFile(str(wheel.path)) as archive:
        try:
            payload = archive.read(resource)
        except KeyError:
            raise ValueError("%s is missing %s" % (wheel.path.name, resource))
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("invalid capability resource %s: %s" % (resource, error))
    if not isinstance(value, dict):
        raise ValueError("capability resource must be a JSON object: %s" % resource)
    tools = value.get("tools")
    if not isinstance(tools, list):
        raise ValueError("capability resource tools must be a list: %s" % resource)
    return payload


def _validate_capability_resources(pairs):
    result = {}
    for project, resource in _CAPABILITY_RESOURCES.items():
        if project not in pairs:
            continue
        wheel = pairs[project][0]
        result[project] = (resource, _wheel_resource(wheel, resource))
    return result


def _venv_python(environment):
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _site_packages(python):
    code = "import json, site; print(json.dumps(site.getsitepackages()))"
    result = subprocess.check_output([str(python), "-c", code], universal_newlines=True)
    paths = json.loads(result.strip())
    if not isinstance(paths, list) or not paths:
        raise ValueError("venv did not report site-packages")
    return Path(paths[0])


def _install_wheels(environment, wheels):
    venv.create(str(environment), with_pip=True, system_site_packages=False)
    python = _venv_python(environment)
    for wheel in wheels:
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel.path)]
        )
    return python


def _validate_installed_resources(site_packages, resources):
    for project, value in resources.items():
        resource, expected = value
        path = site_packages / resource
        if not path.is_file():
            raise ValueError("installed resource missing for %s: %s" % (project, resource))
        if path.read_bytes() != expected:
            raise ValueError("installed resource content mismatch for %s" % project)


def _validate_install_isolation(pairs, resources):
    required = ("linktools", "linktools-common", "linktools-mobile")
    if not all(project in pairs for project in required):
        return
    core = pairs["linktools"][0]
    common = pairs["linktools-common"][0]
    mobile = pairs["linktools-mobile"][0]
    orders = (
        ("core-first", (core, common, mobile)),
        ("mobile-first", (mobile, common, core)),
    )
    with tempfile.TemporaryDirectory(prefix="linktools-wheel-check-") as temporary:
        root = Path(temporary)
        first_python = None
        first_site_packages = None
        for label, order in orders:
            environment = root / label
            python = _install_wheels(environment, order)
            site_packages = _site_packages(python)
            _validate_installed_resources(site_packages, resources)
            if label == "core-first":
                first_python = python
                first_site_packages = site_packages
        if first_python is None or first_site_packages is None:
            raise ValueError("core-first verification environment was not created")
        subprocess.check_call([str(first_python), "-m", "pip", "uninstall", "-y", "linktools-mobile"])
        mobile_resource = first_site_packages / resources["linktools-mobile"][0]
        if mobile_resource.exists():
            raise ValueError("mobile resource remains after uninstall")
        for project in ("linktools", "linktools-common"):
            resource, expected = resources[project]
            path = first_site_packages / resource
            if not path.is_file() or path.read_bytes() != expected:
                raise ValueError("%s resource changed after mobile uninstall" % project)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("projects", nargs="*", help="registered projects to verify")
    args = parser.parse_args()
    try:
        project_paths = _project_directories()
        if not project_paths:
            raise ValueError("no LinkTools projects found")
        selected = tuple(_normalize_name(project) for project in args.projects) if args.projects else tuple(project_paths)
        unknown_selected = sorted(set(selected) - set(project_paths))
        if unknown_selected:
            raise ValueError("unknown selected project(s): %s" % ", ".join(unknown_selected))
        artifacts = _load_artifacts(project_paths)
        _validate_full_artifact_set(artifacts, project_paths, selected)
        pairs = _selected_pairs(artifacts, selected, project_paths)
        _validate_version(pairs, project_paths)
        resources = _validate_capability_resources(pairs)
        _validate_install_isolation(pairs, resources)
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print("[-] Artifact verification failed: %s" % error, file=sys.stderr)
        return 1
    print("[+] Artifact verification passed: %s" % ", ".join(selected))
    return 0


if __name__ == "__main__":
    sys.exit(main())
