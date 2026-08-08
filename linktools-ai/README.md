# linktools-ai

`linktools-ai` is the agent runtime for local coding and durable service
profiles. Its public storage abstraction is `AssetStore`; specification DTOs
and codecs live in `linktools.ai.spec`, while the concrete SQL asset backend is
available from `linktools.ai.asset.sql`.

The three execution profiles are:

- `production-service`: durable Temporal workflows with explicitly assembled
  self-hosted storage and runtime services;
- `production-sandboxed`: blocked until the official Harness wheel provides
  the required Modal capability;
- `local-coding`: direct local execution with project, file and shell tools.

Storage builders are lazy. Callers explicitly initialize the SQL schema and
then construct the storage composition and asset store. The `ai run` command
streams model text, thinking and tool activity; `ai acp` serves the local ACP
transport when its optional dependency is installed.

For local execution, `--project` selects the working directory and `--storage`
selects the Runtime state directory:

```bash
ai-run --project /workspace/project --storage /var/lib/linktools-ai "hello"
ai-acp --project /workspace/project --storage /var/lib/linktools-ai
```

Sessions and execution records are stored below `<storage>/.linktools/`;
tools and agent files remain rooted at the project directory.

For database-backed downstream services, use the public store configuration and
keep the namespace stable for the deployment:

```python
import os

from linktools.ai import RuntimeStoreConfig, open_runtime_store

config = RuntimeStoreConfig.postgresql(
    os.environ["LINKTOOLS_DATABASE_URL"],
    namespace="tenant-id",
    deployment_id="runtime-prod",
)

async with open_runtime_store(config) as stores:
    # Pass stores.domain into the application service composition.
    await stores.domain.sessions.list(tenant_id="tenant-id")
```

Use `RuntimeStoreConfig.sqlite(path, namespace=..., deployment_id=...)` for a
single-node database. MySQL uses
`RuntimeStoreConfig.mysql("mysql+asyncmy://...", namespace=..., deployment_id=...)`.
SQL drivers are loaded only when the corresponding backend is opened.

See `.docs/linktools-ai-integrated-spec.md` and the checked-in manifests under
`scripts/build/matrix` for the architecture and release contract.
