from pathlib import Path
import re

ROOT = Path('.')

def read(path: str) -> str:
    return (ROOT / path).read_text(encoding='utf-8')

def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding='utf-8')

def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected one match, got {count}')
    return text.replace(old, new, 1)

# Local execution: terminal schema is owned by AgentBinding; execution history seal
# remains exact-binding identity, never agent identity.
path = 'linktools-ai/src/linktools/ai/runtime/_local.py'
text = read(path)
text = replace_once(
    text,
    '        definition: AgentDefinition | None,\n        run_id: str | None,',
    '        binding: AgentBinding | None,\n        run_id: str | None,',
    'same-group terminal binding parameter',
)
text = replace_once(
    text,
    '                    agent_digest=binding.definition.digest,\n',
    '                    binding_digest=current.binding_digest,\n',
    'execution terminal seal exact binding',
)
write(path, text)

# AssetRepository public construction tests: registry is now private implementation detail.
path = 'tests/ai/test_asset_repository.py'
text = read(path)
text = replace_once(
    text,
    '''    registry = AssetTypeRegistry()\n    registry.register(_binding())\n    return store, AssetRepository(store, registry.freeze())\n''',
    '''    return store, AssetRepository(store, (_binding(),))\n''',
    'asset repo helper',
)
text = replace_once(
    text,
    '''    registry = AssetTypeRegistry()\n    registry.register(_multi_directory_binding())\n    repository = AssetRepository(store, registry.freeze())\n''',
    '''    repository = AssetRepository(store, (_multi_directory_binding(),))\n''',
    'multi directory repository',
)
# Two identical single-binding repository constructions remain in different tests.
for label in ('counting repository', 'race repository', 'recovery race repository', 'readonly repository'):
    old = '''    registry = AssetTypeRegistry()\n    registry.register(_binding())\n    repository = AssetRepository(store, registry.freeze())\n'''
    if old not in text:
        raise RuntimeError(f'{label}: pattern missing')
    text = text.replace(old, '''    repository = AssetRepository(store, (_binding(),))\n''', 1)

# Replace the registry-specific validation tests with constructor-level public API tests.
start = text.index('def test_registry_freeze_rejects_overlapping_single_file_layouts() -> None:\n')
end = text.index('\n\n@pytest.mark.parametrize("layout"', start)
text = text[:start] + '''def test_repository_rejects_overlapping_single_file_layouts() -> None:\n    store = AssetStore(StorageOverlay(InMemoryAssetBackend()))\n    with pytest.raises(AIError) as error:\n        AssetRepository(\n            store,\n            (\n                AssetTypeBinding(\n                    "sample",\n                    _Value,\n                    (\n                        AssetVariantBinding("short", SingleFileLayout(".md"), _Codec(), "short", 1),\n                        AssetVariantBinding("long", SingleFileLayout(".agent.md"), _Codec(), "long", 1),\n                    ),\n                    "short",\n                ),\n            ),\n        )\n    assert error.value.code is ErrorCode.ASSET_CODEC_CONFLICT\n\n''' + text[end+2:]

start = text.index('def test_registry_accepts_concrete_protocol_subclass_value_type() -> None:\n')
end = text.index('\n\n@pytest.mark.asyncio\nasync def test_concrete_protocol_subclass_keeps_typed_runtime_exact_type', start)
text = text[:start] + '''def test_repository_accepts_concrete_protocol_subclass_value_type() -> None:\n    store = AssetStore(StorageOverlay(InMemoryAssetBackend()))\n    repository = AssetRepository(\n        store,\n        (\n            AssetTypeBinding(\n                "concrete-protocol",\n                _ConcreteProtocolValue,\n                (AssetVariantBinding("file", SingleFileLayout(""), _ConcreteProtocolCodec(), "concrete-protocol", 1),),\n                "file",\n            ),\n        ),\n    )\n    assert repository.kinds == ("concrete-protocol",)\n\n''' + text[end+2:]

text = replace_once(
    text,
    '''    registry = AssetTypeRegistry()\n    registry.register(\n        AssetTypeBinding(\n            "concrete-protocol-runtime",\n            _ConcreteProtocolValue,\n            (\n                AssetVariantBinding(\n                    "file",\n                    SingleFileLayout(""),\n                    _ConcreteProtocolCodec(),\n                    "concrete-protocol-runtime",\n                    1,\n                ),\n            ),\n            "file",\n        )\n    )\n    repository = AssetRepository(store, registry.freeze())\n''',
    '''    repository = AssetRepository(\n        store,\n        (\n            AssetTypeBinding(\n                "concrete-protocol-runtime",\n                _ConcreteProtocolValue,\n                (\n                    AssetVariantBinding(\n                        "file",\n                        SingleFileLayout(""),\n                        _ConcreteProtocolCodec(),\n                        "concrete-protocol-runtime",\n                        1,\n                    ),\n                ),\n                "file",\n            ),\n        ),\n    )\n''',
    'concrete protocol runtime repository',
)
text = replace_once(
    text,
    '''    registry = AssetTypeRegistry()\n    registry.register(\n        AssetTypeBinding(\n            "wrong",\n            _Value,\n            (AssetVariantBinding("wrong", SingleFileLayout(""), _WrongTypeCodec(), "wrong", 1),),\n            "wrong",\n        )\n    )\n    registry.register(\n        AssetTypeBinding(\n            "broken",\n            _Value,\n            (AssetVariantBinding("broken", SingleFileLayout(""), _BrokenCodec(), "broken", 1),),\n            "broken",\n        )\n    )\n    extra = AssetRepository(store, registry.freeze())\n''',
    '''    extra = AssetRepository(\n        store,\n        (\n            AssetTypeBinding(\n                "wrong",\n                _Value,\n                (AssetVariantBinding("wrong", SingleFileLayout(""), _WrongTypeCodec(), "wrong", 1),),\n                "wrong",\n            ),\n            AssetTypeBinding(\n                "broken",\n                _Value,\n                (AssetVariantBinding("broken", SingleFileLayout(""), _BrokenCodec(), "broken", 1),),\n                "broken",\n            ),\n        ),\n    )\n''',
    'codec error repositories',
)
if 'AssetTypeRegistry' in text:
    raise RuntimeError('public AssetTypeRegistry residue remains in asset repository tests')
write(path, text)

# Workspace regression tests use the AgentHandle public session entry only.
path = 'tests/ai/test_workspace_runtime_regressions.py'
text = read(path)
text = text.replace('created = await runtime.create_session("remember")', 'created = await runtime.agent().create_session("remember")')
text = text.replace('await runtime.create_session("custom-tenant")', 'await runtime.agent().create_session("custom-tenant")')
write(path, text)

# Temporal execution request has exactly one current V1 shape and carries the snapshot once.
path = 'linktools-ai/src/linktools/ai/temporal/_request.py'
text = read(path)
text = text.replace('from collections.abc import Mapping\n', 'from collections.abc import Mapping\n')
start = text.index('_EXECUTION_LEGACY_V1_FIELDS = frozenset(')
end = text.index('_logger = environ.get_logger', start)
text = text[:start] + '''_EXECUTION_V1_FIELDS = frozenset(\n    {\n        "version",\n        "user_prompt",\n        "principal",\n        "idempotency_key",\n        "memory_scope",\n        "planning",\n        "thinking",\n        "binding",\n    }\n)\n''' + text[end:]

start = text.index('async def put_execution_request(')
end = text.index('\n\nasync def read_execution_request(', start)
text = text[:start] + '''async def put_execution_request(\n    store: ObjectStore,\n    key_factory: RuntimeObjectKeyFactory,\n    request: ExecutionRequest,\n    *,\n    binding: AgentBindingSnapshot,\n) -> str:\n    if not isinstance(binding, AgentBindingSnapshot):\n        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n    payload: dict[str, JsonValue] = {\n        "version": 1,\n        "user_prompt": request.user_prompt,\n        "principal": _principal_payload(request.principal),\n        "idempotency_key": request.idempotency_key,\n        "memory_scope": request.memory_scope,\n        "planning": request.planning,\n        "thinking": request.thinking,\n        "binding": binding.to_payload(),\n    }\n    reference = await put_runtime_object(\n        store,\n        key_factory,\n        RuntimeDomain.TASK,\n        request.principal.tenant_id,\n        canonical_json_bytes(payload),\n    )\n    _logger.debug(\n        "execution request persisted: tenant=%s request_ref=%s binding=%s",\n        request.principal.tenant_id,\n        reference.key,\n        binding.binding_digest,\n    )\n    return reference.key\n''' + text[end:]

start = text.index('async def read_execution_request(')
end = text.index('\n\ndef _principal_payload(', start)
text = text[:start] + '''async def read_execution_request(\n    store: ObjectStore,\n    key_factory: RuntimeObjectKeyFactory,\n    *,\n    tenant_id: str,\n    request_ref: str,\n) -> ExecutionRequest:\n    request, _binding = await _read_execution_transport(\n        store,\n        key_factory,\n        tenant_id=tenant_id,\n        request_ref=request_ref,\n    )\n    return request\n\n\nasync def load_execution_request(\n    store: ObjectStore,\n    *,\n    namespace: str,\n    state: ExecutionWorkflowState,\n) -> tuple[ExecutionRequest, AgentBindingSnapshot]:\n    if not isinstance(namespace, str) or not namespace.strip():\n        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n    request, binding = await _read_execution_transport(\n        store,\n        RuntimeObjectKeyFactory(namespace),\n        tenant_id=state.tenant_id,\n        request_ref=state.request_ref,\n    )\n    if (\n        request.principal.tenant_id != state.tenant_id\n        or binding.binding_digest != state.binding_digest\n    ):\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    _logger.debug(\n        "execution request loaded: execution=%s request_ref=%s",\n        state.execution_id,\n        state.request_ref,\n    )\n    return request, binding\n\n\nasync def _read_execution_transport(\n    store: ObjectStore,\n    key_factory: RuntimeObjectKeyFactory,\n    *,\n    tenant_id: str,\n    request_ref: str,\n) -> tuple[ExecutionRequest, AgentBindingSnapshot]:\n    payload = await _read_payload(\n        store,\n        key_factory,\n        tenant_id=tenant_id,\n        request_ref=request_ref,\n    )\n    try:\n        value = _load_canonical(payload)\n        request, binding = _execution_request_from_payload(value)\n        if request.principal.tenant_id != tenant_id:\n            raise ValueError("execution request tenant does not match its object key")\n        return request, binding\n    except AIError as error:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n''' + text[end:]

start = text.index('def _execution_request_from_payload(')
end = text.index('\n\ndef _principal_from_payload(', start)
text = text[:start] + '''def _execution_request_from_payload(\n    value: Mapping[str, object],\n) -> tuple[ExecutionRequest, AgentBindingSnapshot]:\n    payload = _mapping(value, _EXECUTION_V1_FIELDS)\n    _require_version(payload["version"], 1)\n    planning = payload["planning"]\n    thinking = payload["thinking"]\n    if not isinstance(planning, bool) or not isinstance(thinking, bool):\n        raise ValueError("execution mode fields are invalid")\n    binding = AgentBindingSnapshot.from_payload(payload["binding"])\n    memory_scope = payload["memory_scope"]\n    if memory_scope is not None and not isinstance(memory_scope, str):\n        raise ValueError("execution memory scope is invalid")\n    request = ExecutionRequest(\n        _require_string(payload["user_prompt"]),\n        _principal_from_payload(payload["principal"]),\n        _require_string(payload["idempotency_key"]),\n        memory_scope,\n        planning,\n        thinking,\n    )\n    return request, binding\n''' + text[end:]
# The canonical transport no longer needs a separate digest parser.
text = re.sub(r'\n\ndef _require_digest\(value: object\) -> str:\n    result = _require_string\(value\)\n    if _DIGEST\.fullmatch\(result\) is None:\n        raise ValueError\("request digest field is invalid"\)\n    return result\n', '', text)
if '_EXECUTION_LEGACY_V1_FIELDS' in text or '_EXECUTION_CURRENT_V1_FIELDS' in text:
    raise RuntimeError('Temporal dual V1 residue remains')
write(path, text)

# Workflow gateway derives indexed binding identity from the exact snapshot.
path = 'linktools-ai/src/linktools/ai/temporal/_gateway.py'
text = read(path)
text = replace_once(text, 'from ..core import JsonValue\n', 'from ..agent import AgentBindingSnapshot\nfrom ..core import JsonValue\n', 'gateway binding import')
text = replace_once(
    text,
    '''        *,\n        binding_digest: str,\n        binding: Mapping[str, JsonValue],\n    ) -> ExecutionHandle:\n''',
    '''        *,\n        binding: AgentBindingSnapshot,\n    ) -> ExecutionHandle:\n''',
    'gateway execution signature',
)
text = replace_once(
    text,
    '''            request,\n            binding_digest=binding_digest,\n            binding=binding,\n        )\n        workflow_request = ExecutionWorkflowInput(\n            execution_id=workflow_id,\n            tenant_id=request.principal.tenant_id,\n            binding_digest=binding_digest,\n''',
    '''            request,\n            binding=binding,\n        )\n        workflow_request = ExecutionWorkflowInput(\n            execution_id=workflow_id,\n            tenant_id=request.principal.tenant_id,\n            binding_digest=binding.binding_digest,\n''',
    'gateway execution request persistence',
)
text = text.replace('            binding_digest,\n        )\n        return await self._client.start_workflow(', '            binding.binding_digest,\n        )\n        return await self._client.start_workflow(', 1)
write(path, text)

# Runtime workflow port uses the same single snapshot contract.
path = 'linktools-ai/src/linktools/ai/runtime/service_api.py'
text = read(path)
text = replace_once(text, 'from ..core import (\n', 'from ..agent import AgentBindingSnapshot\nfrom ..core import (\n', 'service api binding import')
text = replace_once(
    text,
    '''        *,\n        binding_digest: str,\n        binding: Mapping[str, JsonValue],\n    ) -> ExecutionHandle: ...\n''',
    '''        *,\n        binding: AgentBindingSnapshot,\n    ) -> ExecutionHandle: ...\n''',
    'workflow gateway protocol',
)
write(path, text)

# Temporal TaskGraph already has the canonical node snapshot; persist that snapshot once.
path = 'linktools-ai/src/linktools/ai/temporal/_task_operation.py'
text = read(path)
text = replace_once(text, 'from ..core import Principal, TaskStatus, canonical_sha256\n', 'from ..agent import AgentBindingSnapshot\nfrom ..core import Principal, TaskStatus, canonical_sha256\n', 'task operation binding import')
text = replace_once(
    text,
    '''        binding = node.input.get("binding")\n        if not isinstance(binding, Mapping):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        request_ref = await put_execution_request(\n            self._request_store,\n            self._request_keys,\n            execution_request,\n            binding_digest=binding_digest,\n            binding=binding,\n        )\n''',
    '''        try:\n            binding = AgentBindingSnapshot.from_payload(node.input.get("binding"))\n        except AIError as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n        if binding.binding_digest != binding_digest:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        request_ref = await put_execution_request(\n            self._request_store,\n            self._request_keys,\n            execution_request,\n            binding=binding,\n        )\n''',
    'task operation canonical execution transport',
)
write(path, text)

# Contract tests follow mandatory recovery snapshot and canonical gateway transport.
path = 'tests/ai/test_contracts.py'
text = read(path)
text = replace_once(
    text,
    '''        version=1,\n        agent_spec=spec,\n        output_type_module=output.value_type.__module__,\n''',
    '''        version=1,\n        agent_spec=spec,\n        agent_digest="b" * 64,\n        output_type_module=output.value_type.__module__,\n''',
    'contract snapshot agent digest',
)
old = '''    recovery_input = RecoveryExecutionInput(\n        user_prompt="prompt",\n        principal_id="principal",\n        principal_kind="user",\n        session_id=None,\n        memory_scope=None,\n        agent_id="default",\n        binding_digest="binding",\n        lineage_kind="RUN",\n        parent_execution_id=None,\n        root_execution_id="execution",\n        source_execution_id=None,\n        base_execution_id=None,\n        conversation_step_run_id=None,\n        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),\n    )\n'''
new = '''    snapshot = _binding_snapshot(digest="c" * 64)\n    recovery_input = RecoveryExecutionInput(\n        user_prompt="prompt",\n        principal_id="principal",\n        principal_kind="user",\n        session_id=None,\n        memory_scope=None,\n        binding_digest=snapshot.binding_digest,\n        lineage_kind="RUN",\n        parent_execution_id=None,\n        root_execution_id="execution",\n        source_execution_id=None,\n        base_execution_id=None,\n        conversation_step_run_id=None,\n        idempotency=RecoveryIdempotencyInput("scope", "key", "request"),\n        planning=False,\n        thinking=False,\n        binding=snapshot,\n    )\n'''
text = replace_once(text, old, new, 'mandatory recovery binding contract test')
text = replace_once(
    text,
    '''        local,\n        binding_digest=snapshot.binding_digest,\n        binding=snapshot.to_payload(),\n    )\n''',
    '''        local,\n        binding=snapshot,\n    )\n''',
    'gateway canonical binding test',
)
write(path, text)

print('closure fixes and canonical Temporal V1 applied')
