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

### Install packages (editable mode)
```bash
# Install all packages in editable mode
python manage.py install --editable

# Install specific packages
python manage.py install --editable linktools linktools-mobile

# Install without build isolation (faster if dependencies already installed)
python manage.py install --editable --no-isolation linktools-mobile
```

### Build packages
```bash
# Build all packages to dist/
python manage.py build

# Build specific package
python manage.py build linktools-mobile
```

### Clean build artifacts
```bash
python manage.py clean
python manage.py clean linktools-mobile
```

### Run a command after install
```bash
# Unified entry point (shows all installed commands)
python3 -m linktools

# Or use installed CLI scripts
at-frida --help
ct-tools apktool -h
```

Sub-package-specific build steps (Frida TypeScript, Android APK) are documented in each sub-package's `AGENTS.md`.

### `manage.py` (Monorepo Management Script)

Project-level build tool (not a Django manage.py). Supports `init`, `install`, `build`, `clean` subcommands. Discovers sub-packages by scanning directories matching `linktools-*`. `VERSION` env var controls the version written to each sub-package's `.version` file during builds.

## Config System

Config priority (highest to lowest): environment variables → cache → private config → global config → default value. Descriptors chain with `|`:

```python
MY_KEY = Config.Alias("ALT_KEY", type=int) | Config.Prompt(cached=True) | 42
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


## Comments — minimal

- Explain only what naming and structure cannot: intent, constraints, protocols, counter-intuitive behavior.
- No external references (plan sections, review items, issue/PR links, process narrative).
- No restating the code; no history, decision debates, or reviewer-facing prose; no boilerplate.
- Prefer a rename or restructure over a comment.
- If deleting the comment leaves the code clear, delete it.
- Don't clean up unrelated old comments, but fix any a current edit makes stale or false.
