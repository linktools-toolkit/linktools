# AGENTS.md (linktools-common)

Package instructions for `linktools-common`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

## Required Rules

- Keep `ct-*` commands thin. Configuration, platform detection, logging, CLI infrastructure, and tool download/extraction belong to the core framework.
- New commands use the core `BaseCommand` / `BaseCommandGroup` framework and normal command discovery.

## Guidance

`linktools-common/src/linktools/commands/common/` contains:

- `ct-env`: environment/config/platform inspection.
- `ct-grep`: grep/ripgrep wrapper with linktools defaults.
- `ct-tools`: install/list/run tools declared in `assets/tools.json`.
