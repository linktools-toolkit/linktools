# CLAUDE.md (linktools core)

Architecture guidance for the core framework package. Shared concerns (monorepo
structure, `manage.py`, config system, code style) live in the
[repo-root CLAUDE.md](../CLAUDE.md).

## Core Framework (`linktools/src/linktools/`)

- **`core/`** — Four main subsystems:
  - `_environ.py` (`environ` singleton) — manages data/temp directories, logging, config access
  - `_config.py` (`Config`) — multi-layer config system with `Property`, `Alias`, `Prompt`, `Confirm`, `Lazy`, `Error` descriptors chained via `|` operator
  - `_tools.py` (`Tools`, `Tool`) — declarative tool definitions from `assets/tools.json`; handles download, extraction, and execution
  - `_capability.py` (`BaseCapability`) — sub-package self-registration with version and path info
- **`cli/`** — CLI framework: `BaseCommand`, `BaseCommandGroup`, `CommandParser` (enhanced `ArgumentParser`). All commands in all sub-packages inherit from these.
- **`types.py`** — `Stoppable` (context manager pattern), `Timeout`, shared TypeVars/sentinels (`T`, `PathType`, `MISSING`)
- **`errors.py`** — error hierarchy (`Error → ConfigError → ToolError → ToolNotFound/ToolNotSupport/ToolExecError`)
- **`platform.py`** — OS/user/network helpers (`get_system`, `get_user`, `get_uid`/`get_gid`, `get_lan_ip`, `wait_process`, etc.)
- **`runtime/`** — `Process`/`popen` (subprocess wrapper), `Reactor` (event loop), `Proxy`/`IterProxy` (lazy proxies), `EventHandlerMixin` (event dispatch mixin)
- **`decorator.py`** — `@singleton`, `@cached_property`, `@try_except`, `@timeoutable`
- **`rich.py`** — terminal UI: logging, progress bars, `prompt`/`confirm`/`choose`

## Sub-package Layout

Each CLI sub-package follows the same pattern under `{pkg}/src/linktools/`:
```
commands/        — CLI command implementations (one file per command)
capabilities/    — registers the sub-package with the core framework (auto-generated via jinja2)
assets/          — static assets: config templates, built JS/APK artifacts
```
`linktools-ai/` is library-first (it has an optional `cli/` but no `commands/` entry-point group).
