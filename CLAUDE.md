# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Monorepo Structure

This is a Python monorepo for mobile security research tools, split into five independent sub-packages:

| Package | Description | Command Prefix | Architecture doc |
|---------|-------------|----------------|------------------|
| `linktools/` | Core framework: CLI infrastructure, environ, config, tool management | (base) | [CLAUDE.md](linktools/CLAUDE.md) |
| `linktools-common/` | Common tools: `ct-env`, `ct-grep`, `ct-tools` | `ct-` | [CLAUDE.md](linktools-common/CLAUDE.md) |
| `linktools-mobile/` | Android (`at-*`) and iOS (`it-*`) device tools | `at-`, `it-` | [CLAUDE.md](linktools-mobile/CLAUDE.md) |
| `linktools-cntr/` | Docker/Podman container management (`ct-cntr`) | `ct-cntr` | [CLAUDE.md](linktools-cntr/CLAUDE.md) |
| `linktools-ai/` | AI agent runtime: session/execution/swarm on pydantic-ai | — | [CLAUDE.md](linktools-ai/CLAUDE.md) |

Each sub-package lives under `{name}/src/linktools/` and extends the core framework through Python entry points. Each has its own `CLAUDE.md` covering its architecture; this file covers the shared concerns below.

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

Sub-package-specific build steps (Frida TypeScript, Android APK) are documented in each sub-package's `CLAUDE.md`.

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
