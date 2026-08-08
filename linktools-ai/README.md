# linktools-ai

`linktools-ai` is the agent runtime for local workspace execution and durable
service integrations. Runtime topology is selected by the concrete launcher,
workflow gateway and persistence dependencies supplied by the application; the
runtime does not expose an execution profile or deployment category.

Its public storage abstraction is `AssetStore`; specification DTOs and codecs
live in `linktools.ai.spec`, while the concrete SQL asset backend is available
as `linktools.ai.asset.SqlAssetBackend`.

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

from linktools.ai import RuntimePersistenceConfig, open_runtime_resources

config = RuntimePersistenceConfig.postgresql(
    os.environ["LINKTOOLS_DATABASE_URL"],
    namespace="tenant-id",
    deployment_id="runtime-prod",
)

async with open_runtime_resources(config) as resources:
    # Pass resources.domain into the application service composition.
    await resources.domain.sessions.list(tenant_id="tenant-id")
```

Use `RuntimePersistenceConfig.sqlite(path, namespace=..., deployment_id=...)` for a
single-node database. MySQL uses
`RuntimePersistenceConfig.mysql("mysql+asyncmy://...", namespace=..., deployment_id=...)`.
SQL drivers are loaded only when the corresponding backend is opened.

The checked-in architecture and release contract is maintained under
`scripts/build/matrix`. For profile-removal upgrades, drain unfinished affected
start/create requests and snapshot FILE/SQL durable state first. If rollback to
the old binary is required after a durable mutation, restore that snapshot
before rollback; no migration framework is introduced.
