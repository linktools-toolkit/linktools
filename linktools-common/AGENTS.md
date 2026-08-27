# AGENTS.md (linktools-common)

Package instructions for `linktools-common`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

## Required Rules

- Common commands delegate shared configuration, platform, logging, CLI, and tool-management behavior to the core framework instead of duplicating it locally.

## Guidance

`linktools-common/src/linktools/commands/common/` currently contains:

- `ct-env`: environment/config/platform inspection.
- `ct-grep`: grep/ripgrep wrapper with linktools defaults.
- `ct-tools`: install/list/run tools declared in `assets/tools.json`.

New commands normally use the core `BaseCommand` / `BaseCommandGroup` framework and existing command discovery.
