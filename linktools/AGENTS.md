# AGENTS.md (linktools core)

Package instructions for the core framework. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

## Required Rules

- Downstream commands use the core CLI, environment, configuration, logging, and tool-management infrastructure instead of reimplementing it.
- CLI-oriented packages use the established `commands/`, `capabilities/`, and `assets/` roles; do not add a second registration or command-discovery mechanism.
- `linktools-ai` is library-first and is not required to follow the normal CLI package layout.

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

Typical CLI package layout:

```text
commands/        # CLI implementations
capabilities/    # package registration
assets/          # static/generated assets
```
