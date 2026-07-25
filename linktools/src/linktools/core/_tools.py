"""Tool definitions, runtime resolution, and transactional installation."""

import datetime
import json
import os
import re
import shutil
import stat
import uuid
from pathlib import Path

from linktools import utils
from linktools.decorator import cached_property, timeoutable
from linktools.errors import ToolDefinitionError, ToolExecError, ToolInstallError
from linktools.errors import ToolNotFound, ToolNotSupport
from linktools.runtime import popen
from linktools.system import CommandStub, get_interpreter, get_interpreter_ident


_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SPEC_KEYS = {
    "install": {"url", "sha256", "size", "extract_dir", "entrypoint"},
    "run": {"lookup", "path", "runner", "args", "environment"},
}
_TOP_KEYS = {"version", "dependencies", "install", "run", "variables", "variants"}
_MISSING = object()


def _error(name, source, message):
    context = "tool %s" % name
    if source:
        context += " (%s)" % source
    raise ToolDefinitionError("%s: %s" % (context, message))


def _mapping(value, name, source, field):
    if value is None:
        return {}
    if not isinstance(value, dict):
        _error(name, source, "%s must be a mapping" % field)
    return value


def _deep_merge(left, right):
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _tool_payload(payload, owner, source):
    if (not isinstance(payload, dict) or set(payload) - {"schema", "templates", "tools"} or
            payload.get("schema") != 1 or not isinstance(payload.get("tools"), dict)):
        raise ToolDefinitionError("tool configuration: invalid tools schema for %s at %s" % (owner, source))
    return payload["tools"]


def _merge_tool_payload(definitions, sources, tools, source):
    for name, value in tools.items():
        if name in definitions:
            raise ToolDefinitionError("tool configuration: duplicate tool %s from %s and %s" %
                                       (name, sources[name], source))
        definitions[name] = value
        sources[name] = source


def _validate_user_tool_payload(payload, source, existing_names=()):
    tools = _tool_payload(payload, "user tools", source)
    for name, value in tools.items():
        if name not in existing_names and isinstance(value, dict) and not ({"install", "run"} & set(value)):
            raise ToolDefinitionError("tool configuration: user tool %s in %s is not a complete definition" %
                                       (name, source))
    return tools


def _relative(value, name, source, field):
    if value is None:
        return None
    if not isinstance(value, str) or not value or os.path.isabs(value):
        _error(name, source, "%s must be a relative path" % field)
    path = Path(value)
    if ".." in path.parts:
        _error(name, source, "%s escapes its install directory" % field)
    return value


def _rendered_relative(value, name, source, field):
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        _error(name, source, "%s escapes its install directory" % field)
    return value


class InstallSpec(object):
    def __init__(self, url=None, sha256=None, size=None, extract_dir=None, entrypoint=None,
                 name="<unknown>", source=None):
        self.url = url
        self.sha256 = sha256
        self.size = size
        self.extract_dir = _relative(extract_dir, name, source, "install.extract_dir")
        self.entrypoint = _relative(entrypoint, name, source, "install.entrypoint")
        if self.url is not None and not isinstance(self.url, str):
            _error(name, source, "install.url must be a string")
        if self.sha256 is not None and not isinstance(self.sha256, str):
            _error(name, source, "install.sha256 must be a string")
        if self.size is not None and (not isinstance(self.size, int) or self.size < 0):
            _error(name, source, "install.size must be a non-negative integer")

    @classmethod
    def from_mapping(cls, data=None, name="<unknown>", source=None):
        data = _mapping(data, name, source, "install")
        unknown = set(data) - _SPEC_KEYS["install"]
        if unknown:
            _error(name, source, "unknown install fields: %s" % sorted(unknown))
        return cls(name=name, source=source, **data)


class RunSpec(object):
    def __init__(self, lookup=_MISSING, path=None, runner=None, args=(), environment=None,
                 name="<unknown>", source=None):
        self.lookup = lookup
        self.path = path
        self.runner = runner
        self.args = args
        self.environment = {} if environment is None else environment
        if self.lookup is not _MISSING and self.lookup is not None and not isinstance(self.lookup, str):
            _error(name, source, "run.lookup must be a string or null")
        if self.path is not None and not isinstance(self.path, str):
            _error(name, source, "run.path must be a string")
        if self.runner is not None and not isinstance(self.runner, str):
            _error(name, source, "run.runner must be a string")
        if not isinstance(self.args, (list, tuple)):
            _error(name, source, "run.args must be an array")
        if not isinstance(self.environment, dict) or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in self.environment.items()):
            _error(name, source, "run.environment must be a string mapping")
        if any(not isinstance(item, str) for item in self.args):
            _error(name, source, "run.args must be an array of strings")
        self.args = tuple(self.args)
        self.environment = dict(self.environment)

    @classmethod
    def from_mapping(cls, data=None, name="<unknown>", source=None):
        data = _mapping(data, name, source, "run")
        unknown = set(data) - _SPEC_KEYS["run"]
        if unknown:
            _error(name, source, "unknown run fields: %s" % sorted(unknown))
        return cls(name=name, source=source, **data)


class ToolDefinition(object):
    def __init__(self, name, version="", dependencies=(), install=None, run=None,
                 variables=None, variants=None, source=None):
        if not isinstance(name, str) or not _NAME.match(name):
            _error(name, source, "invalid tool name")
        if not isinstance(version, str):
            _error(name, source, "version must be a string")
        if not isinstance(dependencies, (list, tuple)) or any(not isinstance(x, str) for x in dependencies):
            _error(name, source, "dependencies must be an array of strings")
        if variables is not None and not isinstance(variables, dict):
            _error(name, source, "variables must be a mapping")
        self.name, self.version, self.source = name, version, source
        self.dependencies = tuple(dependencies)
        self.install = install if isinstance(install, InstallSpec) else InstallSpec.from_mapping(install, name, source)
        self.run = run if isinstance(run, RunSpec) else RunSpec.from_mapping(run, name, source)
        self.variables = dict(variables or {})
        self.variants = tuple(variants or ())

    @classmethod
    def from_mapping(cls, name, data, source=None, platform="linux", architecture="x86_64"):
        if not isinstance(name, str) or not _NAME.match(name):
            _error(name, source, "invalid tool name")
        data = _mapping(data, name, source, "tool definition")
        unknown = set(data) - _TOP_KEYS
        if unknown:
            _error(name, source, "unknown fields: %s" % sorted(unknown))
        version = data.get("version", "")
        if not isinstance(version, str):
            _error(name, source, "version must be a string")
        deps = data.get("dependencies", ())
        if not isinstance(deps, (list, tuple)) or any(not isinstance(x, str) for x in deps):
            _error(name, source, "dependencies must be an array of strings")
        variables = data.get("variables", {})
        if not isinstance(variables, dict):
            _error(name, source, "variables must be a mapping")
        variants = data.get("variants", ())
        selected = []
        if variants:
            if not isinstance(variants, (list, tuple)):
                _error(name, source, "variants must be an array")
            for variant in variants:
                if not isinstance(variant, dict) or set(variant) - {"match", "dependencies", "install", "run", "variables"}:
                    _error(name, source, "invalid variant fields")
                match = variant.get("match", {})
                if not isinstance(match, dict) or set(match) - {"platform", "architecture"}:
                    _error(name, source, "variant.match only allows platform and architecture")
                variant_deps = variant.get("dependencies", ())
                if not isinstance(variant_deps, (list, tuple)) or any(not isinstance(x, str) for x in variant_deps):
                    _error(name, source, "variant.dependencies must be an array of strings")
                if not isinstance(variant.get("variables", {}), dict):
                    _error(name, source, "variant.variables must be a mapping")
                InstallSpec.from_mapping(variant.get("install"), name, source)
                RunSpec.from_mapping(variant.get("run"), name, source)
                if all(_matches(match.get(k), v) for k, v in (("platform", platform), ("architecture", architecture))):
                    selected.append(variant)
        if len(selected) > 1:
            _error(name, source, "multiple variants match the current platform")
        if selected:
            data = _deep_merge(data, {k: v for k, v in selected[0].items() if k != "match"})
        return cls(name, version=data.get("version", version), dependencies=data.get("dependencies", deps),
                   install=InstallSpec.from_mapping(data.get("install"), name, source),
                   run=RunSpec.from_mapping(data.get("run"), name, source), variables=data.get("variables", variables),
                   variants=variants, source=source)


def _matches(value, actual):
    return value is None or actual in value if isinstance(value, (list, tuple)) else value is None or value == actual


def get_tool_stub_path(environ):
    return environ.get_data_path("scripts", get_interpreter_ident(), "tools_v%s" % environ.version)


class _ToolTemplateView(object):
    def __init__(self, tool):
        self.artifact_path = str(tool.artifact_path)
        self.variables = tool.variables


class _ToolsTemplateView(object):
    def __init__(self, tools):
        self._tools = tools

    def __getitem__(self, name):
        return _ToolTemplateView(self._tools[name])


class Tool(object):
    def __init__(self, tools, definition, version=None):
        self._tools, self.definition = tools, definition
        self.name = definition.name
        self.version = definition.version if version is None else version
        self.platform, self.architecture = tools.environ.system, tools.environ.machine
        self.dependencies = definition.dependencies
        self.variables = dict(definition.variables)
        self._resolve()

    def _format(self, value, field):
        if value is None:
            return None
        context = {"name": self.name, "version": self.version, "platform": self.platform,
                   "architecture": self.architecture, "variables": self.variables,
                   "install_dir": str(getattr(self, "install_dir", "")),
                   "content_dir": str(getattr(self, "content_dir", "")),
                   "artifact_path": str(getattr(self, "artifact_path", "")),
                   "tools": _ToolsTemplateView(self._tools)}
        try:
            return value.format(**context)
        except (KeyError, ValueError, IndexError, AttributeError) as exc:
            raise ToolDefinitionError("tool %s field %s (%s): %s" % (self.name, field, self.definition.source, exc))

    def _resolve(self):
        install = self.definition.install
        self.install_dir = Path(self._tools.environ.get_data_path(
            "tools", self.name, "versions", "%s-%s-%s" % (self.version or "unknown", self.platform, self.architecture)))
        self._url = self._format(install.url, "install.url")
        extract = _rendered_relative(self._format(install.extract_dir, "install.extract_dir"), self.name,
                                     self.definition.source, "install.extract_dir")
        entry = _rendered_relative(self._format(install.entrypoint, "install.entrypoint"), self.name,
                                   self.definition.source, "install.entrypoint")
        self.content_dir = self.install_dir / extract if extract else self.install_dir
        self.variables = {key: self._format(value, "variables.%s" % key)
                          for key, value in self.variables.items()}
        self.artifact_path = self.content_dir / entry if entry else self.content_dir / (utils.guess_file_name(self._url) if self._url else self.name)
        self._lookup = self.name if install and self.definition.run.lookup is _MISSING else self.definition.run.lookup
        self._path = self._format(self.definition.run.path, "run.path")
        self._args = tuple(self._format(x, "run.args") for x in self.definition.run.args)
        self.environment = {k: self._format(v, "run.environment") for k, v in self.definition.run.environment.items()}
        self._runner = self.definition.run.runner
        self._lookup_path = shutil.which(self._lookup) if self._lookup else None
        if self._lookup_path:
            try:
                if os.path.samefile(self._lookup_path, str(self._tools.stub_path / self.name)):
                    self._lookup_path = None
            except (FileNotFoundError, OSError):
                pass
        if self._lookup_path:
            self.command_path = self._lookup_path
        elif self._runner:
            self.command_path = None
        else:
            self.command_path = self._path or (str(self.artifact_path) if self.artifact_path.is_file() else None)
        self.argv = self._make_argv()

    def _make_argv(self):
        user = list(self._args)
        if self.command_path:
            return [self.command_path] + user
        if self._runner:
            return list(self._tools[self._runner].argv) + user
        return [str(self._path or self.artifact_path)] + user

    @property
    def supported(self):
        install = self.definition.install
        archive = bool(install.extract_dir)
        managed = bool(self._url) and (not archive or bool(install.entrypoint))
        return bool(self._lookup_path or self._path or managed)

    @property
    def exists(self):
        return bool(self._lookup_path or (self._path and os.path.isfile(self._path)) or self.artifact_path.is_file())

    def get_variable(self, name, default=None):
        return self.variables.get(name, default)

    def copy(self, **overrides):
        unknown = set(overrides) - {"version"}
        if unknown:
            raise TypeError("only version may be overridden: %s" % sorted(unknown))
        return Tool(self._tools, self.definition, version=overrides.get("version", self.version))

    def prepare(self):
        if not self.supported:
            raise ToolNotSupport("%s is not supported on %s (%s)" % (self.name, self.platform, self.architecture))
        for dependency in self.dependencies:
            self._tools[dependency].prepare()
        if self._runner:
            self._tools[self._runner].prepare()
        if self._lookup_path:
            return
        if self._path:
            if not os.path.isfile(self._path):
                raise ToolNotFound("explicit path is not a regular file: %s" % self._path)
            return
        if not self._tools.installer.is_complete(self):
            self._tools.installer.install(self)
        if not self.artifact_path.is_file():
            raise ToolNotFound(self.name)
        if not self._runner and os.name != "nt":
            self.artifact_path.chmod(self.artifact_path.stat().st_mode | stat.S_IXUSR)
        stub = self._stub
        if not stub.exists:
            stub.write(self.make_cmdargs())
        self._resolve()

    def clear(self):
        if self.install_dir.exists():
            shutil.rmtree(str(self.install_dir))
        if self._stub.exists:
            self._stub.remove()
        active = self._tools.environ.get_data_path("tools", self.name, "active.json")
        if active.exists():
            active.unlink()

    @property
    def _stub(self):
        return CommandStub(self._tools.stub_path, self.name, system=self.platform)

    def make_cmdargs(self):
        from linktools.cli import env
        return [get_interpreter(), "-m", env.__name__, "tool", self.name]

    def popen(self, *args, **kwargs):
        self.prepare()
        append = self._runtime_environment()
        append.update(kwargs.pop("append_env", {}) or {})
        return popen(*(self.argv + list(args)), append_env=append or None, **kwargs)

    def _runtime_environment(self):
        if self._runner:
            result = self._tools[self._runner]._runtime_environment()
            result.update(self.environment)
            return result
        return dict(self.environment)

    @timeoutable
    def exec(self, *args, timeout=None, ignore_errors=False, on_stdout=None, on_stderr=None, error_type=ToolExecError):
        return self.popen(*args, capture_output=True).exec(timeout=timeout, ignore_errors=ignore_errors,
                                                          on_stdout=on_stdout, on_stderr=on_stderr, error_type=error_type)

    def __repr__(self):
        return "Tool<%s>" % self.name


class ToolInstaller(object):
    def __init__(self, environ, base_dir):
        self.environ, self.base_dir = environ, Path(base_dir)

    def is_complete(self, tool):
        target = tool.install_dir
        manifest_path = target / "manifest.json"
        if not tool.artifact_path.is_file() or not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if (manifest.get("schema"), manifest.get("name"), manifest.get("version"),
                manifest.get("platform"), manifest.get("architecture")) != (
                    2, tool.name, tool.version, tool.platform, tool.architecture):
            return False
        try:
            entry = Path(manifest["entrypoint"])
            if entry.is_absolute() or ".." in entry.parts:
                return False
            if entry != tool.artifact_path.relative_to(tool.install_dir):
                return False
        except (KeyError, ValueError, TypeError):
            return False
        files = manifest.get("files", ())
        if not isinstance(files, list):
            return False
        for item in files:
            if not isinstance(item, str):
                return False
            path = Path(item)
            if path.is_absolute() or ".." in path.parts:
                return False
            if not (target / path).is_file():
                return False
        return True

    def install(self, tool):
        from ._download import DownloadRequest
        if not tool._url:
            raise ToolNotSupport("no install source for %s" % tool.name)
        lock = getattr(self.environ, "locks", None)
        manager = lock.process_lock("tool:" + tool.name) if lock else _NullLock()
        with manager:
            target = tool.install_dir
            if self.is_complete(tool):
                return
            if target.exists():
                quarantine = target.parent / ".corrupt" / (target.name + "-" + uuid.uuid4().hex)
                quarantine.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(target), str(quarantine))
            staging = self.base_dir / tool.name / (".staging-" + uuid.uuid4().hex)
            download_dir = Path(self.environ.get_temp_path("tools", "downloads", uuid.uuid4().hex))
            staging.mkdir(parents=True, exist_ok=True)
            download_dir.mkdir(parents=True, exist_ok=True)
            try:
                archive = download_dir / utils.guess_file_name(tool._url)
                self.environ.downloads.download(DownloadRequest(url=tool._url, destination=str(archive),
                    sha256=tool.definition.install.sha256, size=tool.definition.install.size))
                download_size = archive.stat().st_size
                if tool.definition.install.extract_dir:
                    content = staging / tool._format(tool.definition.install.extract_dir, "install.extract_dir")
                    if not utils.is_sub_path(str(content), str(staging)):
                        raise ToolInstallError("extract_dir escapes install directory for %s" % tool.name)
                    content.mkdir(parents=True, exist_ok=True)
                    utils.safe_extract(str(archive), str(content))
                else:
                    content = staging
                    destination = content / tool._format(tool.definition.install.entrypoint, "install.entrypoint") \
                        if tool.definition.install.entrypoint else content / archive.name
                    if not utils.is_sub_path(str(destination), str(staging)):
                        raise ToolInstallError("entrypoint escapes install directory for %s" % tool.name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(archive), str(destination))
                entry = tool._format(tool.definition.install.entrypoint, "install.entrypoint")
                artifact = (staging / tool._format(tool.definition.install.extract_dir, "install.extract_dir") if tool.definition.install.extract_dir else staging) / (entry or archive.name)
                if not artifact.is_file() or not utils.is_sub_path(str(artifact), str(staging)):
                    raise ToolInstallError("entrypoint missing for %s" % tool.name)
                files = sorted(p.relative_to(staging).as_posix() for p in staging.rglob("*") if p.is_file())
                manifest = {"schema": 2, "name": tool.name, "version": tool.version,
                            "platform": tool.platform, "architecture": tool.architecture,
                            "source_url": tool._url, "sha256": tool.definition.install.sha256,
                            "size": download_size,
                            "entrypoint": artifact.relative_to(staging).as_posix(),
                            "files": files, "installed_at": _now_iso()}
                (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(staging), str(target))
                active = self.base_dir / tool.name / "active.json"
                active.parent.mkdir(parents=True, exist_ok=True)
                utils.atomic_write(str(active), json.dumps({"version": tool.version}))
            finally:
                shutil.rmtree(str(staging), ignore_errors=True)
                shutil.rmtree(str(download_dir), ignore_errors=True)


class _NullLock(object):
    def __enter__(self): return self
    def __exit__(self, *args): pass


class Tools(object):
    def __init__(self, environ, config=None, sources=None):
        self.environ = environ
        self.logger = environ.get_logger("tools")
        self.config = environ.build_config("main", "")
        raw = config or {}
        raw = raw.get("tools", raw) if isinstance(raw, dict) else {}
        self.all = {"shell": Tool(self, ToolDefinition("shell", run=RunSpec.from_mapping({"lookup": None, "path": os.environ.get("SHELL") or "/bin/sh"}))),
                    "python": Tool(self, ToolDefinition("python", run=RunSpec.from_mapping({"lookup": None, "path": get_interpreter()})))}
        sources = sources or {}
        definitions = {}
        for name, data in raw.items():
            if name in self.all:
                source = sources.get(name)
                raise ToolDefinitionError("tool %s%s is reserved by the built-in definitions" %
                                           (name, " (%s)" % source if source else ""))
            source = sources.get(name)
            data = json.loads(json.dumps(data))
            prefix = name.replace("-", "_").replace(".", "_").upper()
            for key, path in (("version", (prefix + "_VERSION",)),
                              ("install.url", (prefix + "_INSTALL_URL",)),
                              ("run.lookup", (prefix + "_RUN_LOOKUP",))):
                value = _MISSING
                for env_key in path:
                    try:
                        value = self.config.get(env_key, default=_MISSING)
                    except TypeError:
                        value = self.config.get(env_key)
                    if value is not _MISSING and value is not None:
                        break
                if value is not _MISSING and value is not None:
                    target = data
                    parts = key.split(".")
                    for part in parts[:-1]:
                        target = target.setdefault(part, {})
                    target[parts[-1]] = value
            definitions[name] = ToolDefinition.from_mapping(name, data, source, self.environ.system, self.environ.machine)
        self._definitions = definitions
        for name, definition in definitions.items():
            references = tuple(definition.dependencies) + ((definition.run.runner,) if definition.run.runner else ())
            for dependency in references:
                if dependency not in definitions and dependency not in self.all:
                    source = definition.source
                    raise ToolDefinitionError("tool %s%s depends on missing tool %s" %
                                               (name, " (%s)" % source if source else "", dependency))
        for name in definitions:
            self._ensure_tool(name)
        self._validate_dependencies()

    def _ensure_tool(self, name):
        if name not in self.all:
            try:
                definition = self._definitions[name]
            except KeyError:
                raise ToolNotFound("Not found tool %s" % name)
            self.all[name] = Tool(self, definition)
        return self.all[name]

    @cached_property
    def installer(self):
        return ToolInstaller(self.environ, self.environ.get_data_path("tools"))

    @cached_property
    def stub_path(self):
        return get_tool_stub_path(self.environ)

    def _validate_dependencies(self):
        graph = {name: tuple(tool.dependencies) + ((tool.definition.run.runner,) if tool.definition.run.runner else ()) for name, tool in self.all.items()}
        for name, deps in graph.items():
            for dep in deps:
                if dep not in graph:
                    source = self.all[name].definition.source
                    raise ToolDefinitionError("tool %s%s depends on missing tool %s" %
                                               (name, " (%s)" % source if source else "", dep))
                if dep == name:
                    source = self.all[name].definition.source
                    raise ToolDefinitionError("tool %s%s has cyclic dependency: %s -> %s" %
                                               (name, " (%s)" % source if source else "", name, name))
        def visit(name, stack):
            if name in stack:
                source = self.all[name].definition.source
                raise ToolDefinitionError("tool %s%s has cyclic dependency: %s" %
                                           (name, " (%s)" % source if source else "",
                                            " -> ".join(stack[stack.index(name):] + [name])))
            for dep in graph[name]: visit(dep, stack + [name])
        for name in graph: visit(name, [])

    def __getitem__(self, name):
        return self._ensure_tool(name)
    def __getattr__(self, name): return self[name]
    def __iter__(self): return (tool for tool in self.all.values() if tool.supported)
    def keys(self): return (name for name, tool in self.all.items() if tool.supported)
    def values(self): return (tool for tool in self.all.values() if tool.supported)
    def items(self): return ((name, tool) for name, tool in self.all.items() if tool.supported)


def _now_iso():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
