# CLAUDE.md (linktools-common)

Architecture guidance for the common-tools sub-package. Shared concerns (monorepo
structure, `manage.py`, config system, code style) live in the
[repo-root CLAUDE.md](../CLAUDE.md).

## Commands (`linktools-common/src/linktools/commands/common/`)

General-purpose CLI tools (prefix `ct-`), all built on the core CLI framework
(`BaseCommand` / `BaseCommandGroup`):

- **`ct-env`** (`env.py`) — print/inspect the linktools environment (data/temp
  directories, config paths, platform info).
- **`ct-grep`** (`grep.py`) — recursive grep utility (wraps system `grep`/`ripgrep`
  with linktools-aware defaults).
- **`ct-tools`** (`tools.py`) — declarative tool manager: install / list / run the
  external tools declared in `assets/tools.json` (e.g. `ct-tools apktool ...`).

These are thin, self-contained commands; the heavy lifting (config, tool download/
extraction, platform detection) is delegated to the core framework.
