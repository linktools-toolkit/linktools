# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Monorepo Structure

This is a Python monorepo for mobile security research tools, split into four independent sub-packages:

| Package | Description | Command Prefix |
|---------|-------------|----------------|
| `linktools/` | Core framework: CLI infrastructure, environ, config, tool management | (base) |
| `linktools-common/` | Common tools: `ct-env`, `ct-grep`, `ct-tools` | `ct-` |
| `linktools-mobile/` | Android (`at-*`) and iOS (`it-*`) device tools | `at-`, `it-` |
| `linktools-cntr/` | Docker/Podman container management (`ct-cntr`) | `ct-cntr` |

Each sub-package lives under `{name}/src/linktools/` and extends the core framework through Python entry points.

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

### Build Frida scripts (TypeScript → JS)
```bash
cd linktools-mobile/agents/frida
npm install
npm run build
```

### Build Android APK (android-tools.apk)
```bash
cd linktools-mobile/agents/android
./gradlew --no-daemon :tools:buildTools
```

### Run a command after install
```bash
# Unified entry point (shows all installed commands)
python3 -m linktools

# Or use installed CLI scripts
at-frida --help
ct-tools apktool -h
```

## Architecture

### Core Framework (`linktools/src/linktools/`)

- **`core/`** — Four main subsystems:
  - `_environ.py` (`environ` singleton) — manages data/temp directories, logging, config access
  - `_config.py` (`Config`) — multi-layer config system with `Property`, `Alias`, `Prompt`, `Confirm`, `Lazy`, `Error` descriptors chained via `|` operator
  - `_tools.py` (`Tools`, `Tool`) — declarative tool definitions from `assets/tools.json`; handles download, extraction, and execution
  - `_capability.py` (`BaseCapability`) — sub-package self-registration with version and path info
- **`cli/`** — CLI framework: `BaseCommand`, `BaseCommandGroup`, `CommandParser` (enhanced `ArgumentParser`). All commands in all sub-packages inherit from these.
- **`types.py`** — `Stoppable` (context manager pattern), `Reactor` (event loop), `Timeout`, error hierarchy (`Error → ConfigError → ToolError → ToolNotFound/ToolNotSupport/ToolExecError`)
- **`decorator.py`** — `@singleton`, `@cached_property`, `@try_except`, `@timeoutable`
- **`rich.py`** — terminal UI: logging, progress bars, `prompt`/`confirm`/`choose`

### Sub-package Layout

Each sub-package follows the same pattern under `{pkg}/src/linktools/`:
```
commands/        — CLI command implementations (one file per command)
capabilities/    — registers the sub-package with the core framework (auto-generated via jinja2)
assets/          — static assets: config templates, built JS/APK artifacts
```

### Mobile Sub-package (`linktools-mobile/src/linktools/mobile/`)

- **`android/`** — `adb.py` (ADB wrapper with multi-device selection), `types.py` (Android-specific types)
- **`ios/`** — `ios.py` (go-ios wrapper), `ipa.py` (IPA parser), `sib.py`, `types.py`
- **`frida/`** — Frida integration: `app.py` (`FridaApplication`, `FridaSession`, `FridaScript`, `FridaReactor`), `server.py` (`FridaServer`, `FridaAndroidServer`, `FridaIOSServer`), `script.py` (`FridaUserScript`, `FridaEvalCode`, `FridaScriptFile`)

### Frida TypeScript Agents (`linktools-mobile/agents/frida/`)

TypeScript source for the built-in Frida scripts. The compiled output (`frida.js`, `frida-*.js`) is committed to `linktools-mobile/src/linktools/assets/` as a build artifact. Key library in `lib/java.ts` provides `JavaHelper` with `hookMethod`, `hookMethods`, `hookAllMethods`, `bypassSslPinning`, etc.

### Android APK Agent (`linktools-mobile/agents/android/`)

Gradle/Android project that builds `android-tools.apk`. The built APK is committed to `linktools-mobile/src/linktools/assets/android-tools.*` as a build artifact.

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
