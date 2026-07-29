"""The only runtime composition root."""

from ..errors import StorageCapabilityError
from ..execution.persistence.local import LocalExecutionBackend
from ..model.resolver import ModelResolver
from .dependencies import RuntimeDependencies
from .facade import Runtime
from .requirements import RuntimeRequirements, RuntimeTopology
from .storage import RuntimeStorage
from ..execution.query import ExecutionQueryService
from .executor import RuntimeExecutor


def build_runtime(*, storage: RuntimeStorage, dependencies: RuntimeDependencies | None = None, requirements: RuntimeRequirements = RuntimeRequirements(), model_resolver: ModelResolver | None = None) -> Runtime:
    if requirements.topology is RuntimeTopology.MULTI_PROCESS and isinstance(storage.execution.backend, LocalExecutionBackend):
        raise StorageCapabilityError("local execution storage only supports single-process topology")
    dependencies = dependencies or RuntimeDependencies(model_resolver or ModelResolver())
    if requirements.tools and storage.tools is None:
        raise StorageCapabilityError("tools are required but no tool store was configured")
    if requirements.tasks and storage.tasks is None:
        raise StorageCapabilityError("tasks are required but no task store was configured")
    if requirements.memory and storage.memory is None:
        raise StorageCapabilityError("memory is required but no memory store was configured")
    if requirements.artifacts and storage.artifacts is None:
        raise StorageCapabilityError("artifacts are required but no artifact store was configured")
    executor = RuntimeExecutor(storage.execution, dependencies.model_resolver)
    return Runtime(executor, ExecutionQueryService(storage.execution))


__all__ = ["build_runtime"]
