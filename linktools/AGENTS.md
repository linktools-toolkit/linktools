# AGENTS.md (linktools core)

Package instructions for the core framework. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

## Required Rules

- Downstream packages reuse the core CLI, environment, configuration, logging, and tool-management infrastructure instead of duplicating equivalent framework layers.
- Do not introduce a competing registration or command-discovery system alongside the core mechanism.

## Guidance

### Core map (`linktools/src/linktools/`)

| Area | Responsibility |
| --- | --- |
| `core/` | Environment, configuration, tools, capabilities |
| `cli/` | Command framework and parsers |
| `runtime/` | Process, reactor, proxy, event primitives |
| `types.py` / `errors.py` / `platform.py` | Shared types, errors, OS/user/network helpers |
| `decorator.py` | Shared decorators |
| `rich.py` | Terminal logging, progress, prompts |

CLI-oriented packages currently use `commands/`, `capabilities/`, and `assets/`. `linktools-ai` is library-first and does not need to follow that layout.
