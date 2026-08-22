# AGENTS.md

Repository-wide guidance for coding agents. Read the affected package's `AGENTS.md` for package-specific architecture and rules.

## Monorepo Structure

| Package             | Description                                              | Commands       | Architecture                            |
| ------------------- | -------------------------------------------------------- | -------------- | --------------------------------------- |
| `linktools/`        | Core framework: CLI, environ, config, tool management    | base           | [AGENTS.md](linktools/AGENTS.md)        |
| `linktools-common/` | Common tools: `ct-env`, `ct-grep`, `ct-tools`            | `ct-*`         | [AGENTS.md](linktools-common/AGENTS.md) |
| `linktools-mobile/` | Android and iOS device tools                             | `at-*`, `it-*` | [AGENTS.md](linktools-mobile/AGENTS.md) |
| `linktools-cntr/`   | Docker/Podman container management                       | `ct-cntr`      | [AGENTS.md](linktools-cntr/AGENTS.md)   |
| `linktools-ai/`     | AI agent runtime: session/execution/swarm on pydantic-ai | —              | [AGENTS.md](linktools-ai/AGENTS.md)     |

Each package lives under `{name}/src/linktools/` and extends the core framework through entry points.

## Development Commands

`manage.py` is the repository-wide project and release-gate entry point. Commands operate on all registered packages by default or an optional package list. `VERSION` controls the `.version` written at build time.

```bash
python manage.py install --editable
python manage.py install --editable linktools linktools-mobile
python manage.py install --editable --no-isolation linktools-mobile
python manage.py check
python manage.py check linktools-ai
python manage.py build [linktools-mobile]
python manage.py verify [linktools-mobile]
python manage.py clean [linktools-mobile]
```

Repository development, build, and release helper code belongs under root `scripts/`. Package directories must not add their own release-script trees. Project-specific gates belong under `scripts/check/<project>/` only when the project has real project-specific rules.

After installation:

```bash
python3 -m linktools
at-frida --help
ct-tools apktool -h
```

Package-specific generated assets, such as Frida TypeScript and Android APKs, are documented in each package's `AGENTS.md`.

## Entry Points / Plugin Discovery

Packages register commands and capabilities through `[tool.linktools.scripts]` in `pyproject.toml`. Installed plugins are discovered automatically at runtime; no manual registration is required after installation.

## Git Commit Messages

Commit subjects must use `type(scope)`, such as `fix(ai)`, `refactor(core)`, `feat(mobile)`, `docs(common)`, or `test(cntr)`, where scope identifies the affected package or module.

Keep subjects short and describe the actual change, e.g. `fix(ai): release tenant-scoped step history on abort`. Do not use process-oriented subjects such as "fix review gaps", "address review comments", "cleanup", or "final fixes".

## Release / CI

A Python package release uses one fixed gate sequence from the repository root:

```text
prepare generated assets
-> install editable development environment
-> python manage.py check
-> python manage.py build
-> python manage.py verify
-> publish to PyPI
-> commit generated artifacts
-> tag / GitHub Release
```

`check` is a read-only source/test gate and `verify` validates the built artifacts. Publishing or tagging must not run after either command fails. GitHub releases also build the Frida JS bundle and Android APK before the Python gates.

## Python Code Style

* **Compatibility**: `linktools-ai` requires Python ≥3.10; all other packages support Python ≥3.6. Code must remain compatible with the affected package's minimum version.
* **File headers**: every `.py` starts with:

  ```python
  #!/usr/bin/env python3
  # -*- coding: utf-8 -*-
  ```
* **Public API is annotated**: every public function/method (not `_`-prefixed) has parameter and return annotations inferred from its body and call sites. Use `Any` only for genuinely untyped values.
* **Quoting**: in Python 3.6-compatible packages, quote annotations containing `|` or `[...]` (`"float | None"`, `"list[str]"`). Bare primitives (`str`, `int`, `bool`, `Any`, …) and bare class names stay unquoted unless imported under `TYPE_CHECKING`.
* **`TYPE_CHECKING`**: annotation-only imports go under `if TYPE_CHECKING:`; runtime imports stay at module scope. Never move names exported through `__all__` or required by runtime-resolved annotations such as SQLAlchemy `Mapped[...]` or Pydantic fields.
* **Interface boundaries**: use public APIs only. Never access private state through reflection (`getattr(x, "_field")`, `x.__dict__`, name-mangled attributes) or another module's private members. If the public API lacks required data, add a public method and implement it across all affected backends/Protocol implementers.
* **Logging**: use `environ.get_logger(...)` or `environ.logger`, never `logging.getLogger(...)`. Use relative names such as `"ai.execution.service"`; the `linktools.` prefix is added automatically. Log key decisions and transitions. Guard only expensive debug logging with `if environ.debug:`. Use `exc_info` only when the traceback aids diagnosis. Direct `logging.*` is reserved for the core `_logging.py` manager and CLI entry points configuring the root logger.

## Module & Class Structure

* **High cohesion, low coupling**: each file, class, and package owns one primary concern. Keep small, closely related units together; do not create modules solely for a single Protocol, helper, or constants. The test is cohesion, not line count.
* **Behavior ownership**: behavior that belongs to a class should be a method, not a module-level function. Use an instance method when instance state is needed; otherwise prefer `classmethod` when the behavior belongs to the class or its subclasses. Use `staticmethod` only when the behavior belongs to the class namespace but requires neither instance nor class context.
* **Private modules**: `_`-prefixed modules are re-exported through `__init__.py`; consumers import from the package (`from linktools.core import ConfigField`), not the private module. Modules addressed externally by dotted path, entry point, or directory scanning remain public, such as `capabilities/mobile.py` and files under `commands/`.
* **No runtime circular dependencies**: applies at both module and package level. `TYPE_CHECKING`-only references are allowed. Break runtime cycles by extracting shared dependencies into a lower layer or defining a Protocol/interface in a base module. Check every new runtime import for cycles.

## Comments — Minimal

* Comment only intent, constraints, protocols, or counter-intuitive behavior that naming and structure cannot express.
* No external plan/review references, issue/PR links, process narrative, history, decision debates, reviewer-facing prose, boilerplate, or code restatement.
* Prefer clear naming or structure over comments.
* Delete comments that add no information.
* Do not clean up unrelated old comments, but fix comments made stale or false by the current change.
