from pathlib import Path
import re

ROOT = Path('.')

def read(path: str) -> str:
    return (ROOT / path).read_text(encoding='utf-8')

def write(path: str, text: str) -> None:
    (ROOT / path).write_text(text, encoding='utf-8')

def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f'{label}: expected one match, got {count}')
    return text.replace(old, new, 1)

path = 'linktools-ai/src/linktools/ai/temporal/_request.py'
text = read(path)
start = text.index('_EXECUTION_LEGACY_V1_FIELDS = frozenset(')
end = text.index('_logger = environ.get_logger', start)
text = text[:start] + '''_EXECUTION_V1_FIELDS = frozenset(\n    {\n        "version",\n        "user_prompt",\n        "principal",\n        "idempotency_key",\n        "memory_scope",\n        "planning",\n        "thinking",\n        "binding",\n    }\n)\n''' + text[end:]

start = text.index('async def put_execution_request(')
end = text.index('\n\nasync def read_execution_request(', start)
text = text[:start] + '''async def put_execution_request(\n    store: ObjectStore,\n    key_factory: RuntimeObjectKeyFactory,\n    request: ExecutionRequest,\n    *,\n    binding: AgentBindingSnapshot,\n) -> str:\n    if not isinstance(binding, AgentBindingSnapshot):\n        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n    payload: dict[str, JsonValue] = {\n        "version": 1,\n        "user_prompt": request.user_prompt,\n        "principal": _principal_payload(request.principal),\n        "idempotency_key": request.idempotency_key,\n        "memory_scope": request.memory_scope,\n        "planning": request.planning,\n        "thinking": request.thinking,\n        "binding": binding.to_payload(),\n    }\n    reference = await put_runtime_object(\n        store,\n        key_factory,\n        RuntimeDomain.TASK,\n        request.principal.tenant_id,\n        canonical_json_bytes(payload),\n    )\n    _logger.debug(\n        "execution request persisted: tenant=%s request_ref=%s binding=%s",\n        request.principal.tenant_id,\n        reference.key,\n        binding.binding_digest,\n    )\n    return reference.key\n''' + text[end:]

start = text.index('async def read_execution_request(')
end = text.index('\n\ndef _principal_payload(', start)
text = text[:start] + '''async def read_execution_request(\n    store: ObjectStore,\n    key_factory: RuntimeObjectKeyFactory,\n    *,\n    tenant_id: str,\n    request_ref: str,\n) -> ExecutionRequest:\n    request, _binding = await _read_execution_transport(\n        store,\n        key_factory,\n        tenant_id=tenant_id,\n        request_ref=request_ref,\n    )\n    return request\n\n\nasync def load_execution_request(\n    store: ObjectStore,\n    *,\n    namespace: str,\n    state: ExecutionWorkflowState,\n) -> tuple[ExecutionRequest, AgentBindingSnapshot]:\n    if not isinstance(namespace, str) or not namespace.strip():\n        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)\n    request, binding = await _read_execution_transport(\n        store,\n        RuntimeObjectKeyFactory(namespace),\n        tenant_id=state.tenant_id,\n        request_ref=state.request_ref,\n    )\n    if request.principal.tenant_id != state.tenant_id or binding.binding_digest != state.binding_digest:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n    _logger.debug(\n        "execution request loaded: execution=%s request_ref=%s",\n        state.execution_id,\n        state.request_ref,\n    )\n    return request, binding\n\n\nasync def _read_execution_transport(\n    store: ObjectStore,\n    key_factory: RuntimeObjectKeyFactory,\n    *,\n    tenant_id: str,\n    request_ref: str,\n) -> tuple[ExecutionRequest, AgentBindingSnapshot]:\n    payload = await _read_payload(\n        store,\n        key_factory,\n        tenant_id=tenant_id,\n        request_ref=request_ref,\n    )\n    try:\n        value = _load_canonical(payload)\n        request, binding = _execution_request_from_payload(value)\n        if request.principal.tenant_id != tenant_id:\n            raise ValueError("execution request tenant does not match its object key")\n        return request, binding\n    except AIError as error:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:\n        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n''' + text[end:]

start = text.index('def _execution_request_from_payload(')
end = text.index('\n\ndef _principal_from_payload(', start)
text = text[:start] + '''def _execution_request_from_payload(\n    value: Mapping[str, object],\n) -> tuple[ExecutionRequest, AgentBindingSnapshot]:\n    payload = _mapping(value, _EXECUTION_V1_FIELDS)\n    _require_version(payload["version"], 1)\n    planning = payload["planning"]\n    thinking = payload["thinking"]\n    if not isinstance(planning, bool) or not isinstance(thinking, bool):\n        raise ValueError("execution mode fields are invalid")\n    binding = AgentBindingSnapshot.from_payload(payload["binding"])\n    memory_scope = payload["memory_scope"]\n    if memory_scope is not None and not isinstance(memory_scope, str):\n        raise ValueError("execution memory scope is invalid")\n    request = ExecutionRequest(\n        _require_string(payload["user_prompt"]),\n        _principal_from_payload(payload["principal"]),\n        _require_string(payload["idempotency_key"]),\n        memory_scope,\n        planning,\n        thinking,\n    )\n    return request, binding\n''' + text[end:]
text = re.sub(r'\n\ndef _require_digest\(value: object\) -> str:\n    result = _require_string\(value\)\n    if _DIGEST\.fullmatch\(result\) is None:\n        raise ValueError\("request digest field is invalid"\)\n    return result\n', '', text)
if '_EXECUTION_LEGACY_V1_FIELDS' in text or '_EXECUTION_CURRENT_V1_FIELDS' in text:
    raise RuntimeError('Temporal dual V1 residue remains')
write(path, text)

path = 'linktools-ai/src/linktools/ai/temporal/_gateway.py'
text = read(path)
text = once(text, 'from ..core import JsonValue\n', 'from ..agent import AgentBindingSnapshot\nfrom ..core import JsonValue\n', 'gateway binding import')
text = once(text,
    '''        *,\n        binding_digest: str,\n        binding: Mapping[str, JsonValue],\n    ) -> ExecutionHandle:\n''',
    '''        *,\n        binding: AgentBindingSnapshot,\n    ) -> ExecutionHandle:\n''',
    'gateway execution signature')
text = once(text,
    '''            request,\n            binding_digest=binding_digest,\n            binding=binding,\n        )\n        workflow_request = ExecutionWorkflowInput(\n            execution_id=workflow_id,\n            tenant_id=request.principal.tenant_id,\n            binding_digest=binding_digest,\n''',
    '''            request,\n            binding=binding,\n        )\n        workflow_request = ExecutionWorkflowInput(\n            execution_id=workflow_id,\n            tenant_id=request.principal.tenant_id,\n            binding_digest=binding.binding_digest,\n''',
    'gateway canonical request')
text = once(text,
    '''            workflow_id,\n            binding_digest,\n        )\n''',
    '''            workflow_id,\n            binding.binding_digest,\n        )\n''',
    'gateway binding log')
write(path, text)

path = 'linktools-ai/src/linktools/ai/runtime/service_api.py'
text = read(path)
text = once(text, 'from ..core import (\n', 'from ..agent import AgentBindingSnapshot\nfrom ..core import (\n', 'service api binding import')
text = once(text,
    '''        *,\n        binding_digest: str,\n        binding: Mapping[str, JsonValue],\n    ) -> ExecutionHandle: ...\n''',
    '''        *,\n        binding: AgentBindingSnapshot,\n    ) -> ExecutionHandle: ...\n''',
    'workflow gateway protocol')
write(path, text)

path = 'linktools-ai/src/linktools/ai/temporal/_task_operation.py'
text = read(path)
text = once(text, 'from ..core import Principal, TaskStatus, canonical_sha256\n', 'from ..agent import AgentBindingSnapshot\nfrom ..core import Principal, TaskStatus, canonical_sha256\n', 'task operation binding import')
text = once(text,
    '''        binding = node.input.get("binding")\n        if not isinstance(binding, Mapping):\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        request_ref = await put_execution_request(\n            self._request_store,\n            self._request_keys,\n            execution_request,\n            binding_digest=binding_digest,\n            binding=binding,\n        )\n''',
    '''        try:\n            binding = AgentBindingSnapshot.from_payload(node.input.get("binding"))\n        except AIError as error:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error\n        if binding.binding_digest != binding_digest:\n            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)\n        request_ref = await put_execution_request(\n            self._request_store,\n            self._request_keys,\n            execution_request,\n            binding=binding,\n        )\n''',
    'task operation canonical request')
write(path, text)

path = 'tests/ai/test_contracts.py'
text = read(path)
text = once(text,
    '''        local,\n        binding_digest=snapshot.binding_digest,\n        binding=snapshot.to_payload(),\n    )\n''',
    '''        local,\n        binding=snapshot,\n    )\n''',
    'gateway test canonical binding')
write(path, text)

print('canonical Temporal V1 transport applied')
