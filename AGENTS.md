# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Monorepo Structure

This is a Python monorepo for mobile security research tools, split into five independent sub-packages:

| Package | Description | Command Prefix | Architecture doc |
|---------|-------------|----------------|------------------|
| `linktools/` | Core framework: CLI infrastructure, environ, config, tool management | (base) | [AGENTS.md](linktools/AGENTS.md) |
| `linktools-common/` | Common tools: `ct-env`, `ct-grep`, `ct-tools` | `ct-` | [AGENTS.md](linktools-common/AGENTS.md) |
| `linktools-mobile/` | Android (`at-*`) and iOS (`it-*`) device tools | `at-`, `it-` | [AGENTS.md](linktools-mobile/AGENTS.md) |
| `linktools-cntr/` | Docker/Podman container management (`ct-cntr`) | `ct-cntr` | [AGENTS.md](linktools-cntr/AGENTS.md) |
| `linktools-ai/` | AI agent runtime: session/execution/swarm on pydantic-ai | — | [AGENTS.md](linktools-ai/AGENTS.md) |

Each sub-package lives under `{name}/src/linktools/` and extends the core framework through Python entry points. Each has its own `AGENTS.md` covering its architecture; this file covers the shared concerns below.

## Development Commands

`manage.py` is the project-level build tool (not Django's). Its subcommands — `init`, `install`, `build`, `clean` — discover sub-packages by scanning `linktools-*` dirs and each take an optional package list (default: all). `VERSION` env var controls the version written to each sub-package's `.version` file at build time. Sub-package-specific build steps (Frida TypeScript, Android APK) live in each sub-package's `AGENTS.md`.

```bash
python manage.py install --editable                              # all packages, editable
python manage.py install --editable linktools linktools-mobile   # specific packages
python manage.py install --editable --no-isolation linktools-mobile  # skip build isolation (faster if deps present)
python manage.py build [linktools-mobile]                        # build to dist/ (all or one)
python manage.py clean [linktools-mobile]                        # clean artifacts (all or one)
```

After install, the unified entry point lists all installed commands; installed CLI scripts work too:

```bash
python3 -m linktools      # unified entry point
at-frida --help           # or installed CLI scripts
ct-tools apktool -h
```

## Config System

Fields are `ConfigField`s resolved through a `ConfigSource` chain (highest priority first): `EnvironmentSource` → `RuntimeOverrideSource` → `PersistentSource` → `FileSource` → `DictSource` → `DefaultSource`. Within a field, multiple providers chain via `ChainProvider` (tried in order, first non-exception wins, else the field's `default`); `ConfigField.chain(...)` is the shorthand.

```python
from linktools.core import ConfigField, AliasProvider, PromptProvider, LazyProvider

HOST = ConfigField.chain(
    AliasProvider("ALT_KEY"),          # read from an alias env/config key first
    PromptProvider(cached=True),       # then interactively prompt (and cache)
    LazyProvider(lambda: "localhost"),  # then a computed fallback
)                                       # name comes from the configs-dict key
```

## Entry Points / Plugin Discovery

Sub-packages register commands and capabilities via Python entry points declared in `pyproject.toml` under `[tool.linktools.scripts]`. The core framework discovers them at runtime — no manual registration needed after `pip install`.

## Release / CI

On GitHub release: CI builds the Frida JS bundle, Android APK, and Python wheels, then publishes to PyPI. The built artifacts and `.version` files are committed back to the repo automatically.

## Python Code Style (All Sub-packages)

- **Python ≥3.10 minimum** — no `from __future__ import annotations` (annotations are evaluated eagerly, so the quoting rule below is mandatory).
- **File headers**: every `.py` starts with:
  ```python
  #!/usr/bin/env python3
  # -*- coding: utf-8 -*-
  ```
- **Public API is annotated**: every public method/function (not `_`-prefixed) has parameter and return annotations, inferred from body/call sites. Use `Any` only for genuinely untyped values.
- **Quoting**: quote annotations containing `|` or `[...]` (`"float | None"`, `"list[str]"`); bare primitives (`str`, `int`, `bool`, `Any`, …) and bare class names stay unquoted — unless the name is imported under `TYPE_CHECKING`, in which case it must be quoted.
- **`TYPE_CHECKING`**: annotation-only imports go under `if TYPE_CHECKING:`; runtime imports stay at module scope. Never move names in `__all__` or referenced by runtime-resolved annotations (SQLAlchemy `Mapped[...]`, Pydantic fields).
- **Respect interface boundaries**: reach an object's data only through its public API, never by reflection (`getattr(x, "_field")`, `x.__dict__`, name-mangled attrs) or by reaching across layers into another module's privates. If the public surface doesn't expose what you need, add a public method (and implement it on each backend/protocol implementer) rather than tunneling past it. Privates (`_`-prefixed) are implementation details that can change without notice.
- **Logging via `environ`**: use `environ.get_logger(...)` (or `environ.logger`) for loggers, never `logging.getLogger(...)`. Pass a relative name like `"ai.execution.service"` — the `linktools.` prefix is added automatically. Log at key decision/transition points so behavior is observable. Wrap only expensive debug logs (heavy formatting, large payload dumps, tight loops) in `if environ.debug:`; ordinary `logger.debug(...)` calls need no guard. `exc_info` is not required on every error log — set it (`exc_info=environ.debug`, or `exc_info=True` for failures where the traceback is essential to root-cause the bug) only at points where the stack trace is genuinely needed for triage. Direct `logging.*` is reserved for the core `_logging.py` manager and CLI entry points that configure the root logger.

## Module & Class Structure

- **High cohesion, low coupling**: each file, class, and package owns one concern — a file holds one primary abstraction (plus its private helpers), a package holds one subsystem. But don't over-fragment: small closely-related units belong together. Merge a tiny file into its sibling/parent rather than spinning up a new module just to hold a few lines (e.g. a single Protocol, one helper function, a constants-only module). The test is cohesion, not line count: things that change together and are used together live together.
- **No runtime circular dependencies**: applies at the module level and the package level. Module level: if A imports B (module scope), B must not import A at module scope — directly or transitively — or it will fail to load. Package level: it is just as forbidden for some files in package A to depend on package B while some files in B depend back on A, even if no single module pair is directly cyclic. `TYPE_CHECKING` references are exempt: when two objects legitimately hold each other (parent↔child, observer↔subject, coordinator↔worker), annotating each side with the other's name under `if TYPE_CHECKING:` is fine — it doesn't run at import time. Break *runtime* cycles by extracting the shared dependency into a lower layer, or by defining a Protocol/interface in a base module that both depend on (dependency inversion). When adding a runtime import, check it doesn't close a cycle.

## Comments — minimal

- Explain only what naming and structure cannot: intent, constraints, protocols, counter-intuitive behavior.
- No external references (plan sections, review items, issue/PR links, process narrative).
- No restating the code; no history, decision debates, or reviewer-facing prose; no boilerplate.
- Prefer a rename or restructure over a comment.
- If deleting the comment leaves the code clear, delete it.
- Don't clean up unrelated old comments, but fix any a current edit makes stale or false.
