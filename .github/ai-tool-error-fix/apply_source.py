from pathlib import Path

path = Path('linktools-ai/src/linktools/ai/runtime/_capabilities.py')
text = path.read_text()

def replace_once(old: str, new: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'replacement count {count} for:\n{old[:120]}')
    text = text.replace(old, new, 1)

replace_once(
'''_MODEL_USAGE_CACHE_WRITE_METADATA_KEY = "linktools.ai.model_usage.cache_write_tokens"\n_REPOSITORY_MARKER_HEADER = "[linktools.repository-instructions.v1]"''',
'''_MODEL_USAGE_CACHE_WRITE_METADATA_KEY = "linktools.ai.model_usage.cache_write_tokens"\n_MODEL_TOOL_ERROR_MAX_CHARS = 4096\n_MODEL_TOOL_ERROR_HEAD_CHARS = 1024\n_MODEL_TOOL_ERROR_TRUNCATION_MARKER = "...[truncated]..."\n_MODEL_EFFECT_UNKNOWN_MESSAGE = "TOOL_EFFECT_UNKNOWN: verify side effects before retry"\n_MODEL_RETRY_PREFIX = "TOOL_RETRY_REQUIRED"\n_MODEL_FAILED_PREFIX = "TOOL_EXECUTION_FAILED"\n_REPOSITORY_MARKER_HEADER = "[linktools.repository-instructions.v1]"''')

replace_once(
'''        decision = await self._tool_operations.begin(\n            ctx,\n            call,\n            tool_def,\n            args,\n            policy.replay_safe,\n        )''',
'''        try:\n            decision = await self._tool_operations.begin(\n                ctx,\n                call,\n                tool_def,\n                args,\n                policy.replay_safe,\n            )\n        except AIError as error:\n            if error.code is ErrorCode.TOOL_EFFECT_UNKNOWN:\n                raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error\n            raise''')

replace_once(
'''                        await self._mark_unknown(state, heartbeat_error)\n                        raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from heartbeat_error''',
'''                        await self._mark_unknown(state, heartbeat_error)\n                        keep_call_state = False\n                        raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from heartbeat_error''')

replace_once(
'''                await self._mark_unknown(state, error)\n                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error\n            try:\n                await self._fail_known_effect(''',
'''                await self._mark_unknown(state, error)\n                keep_call_state = False\n                raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error\n            try:\n                await self._fail_known_effect(''')

replace_once(
'''                await self._mark_unknown(state, signal)\n                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from signal\n            unsupported = AIError(''',
'''                await self._mark_unknown(state, signal)\n                keep_call_state = False\n                raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from signal\n            unsupported = AIError(''')

replace_once(
'''            await self._mark_unknown(state, error)\n            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error\n        finally:\n            await self._stop_heartbeat(state)''',
'''            await self._mark_unknown(state, error)\n            keep_call_state = False\n            raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error\n        finally:\n            await self._stop_heartbeat(state)''')

replace_once(
'''                    await self._mark_unknown(state, error)\n                    raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error\n                return await self._fail_known_effect(''',
'''                    await self._mark_unknown(state, error)\n                    raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error\n                return await self._fail_known_effect(''')

replace_once(
'''                await self._mark_unknown(state, error)\n                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error\n            state.preserve_started = True''',
'''                await self._mark_unknown(state, error)\n                raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error\n            state.preserve_started = True''')

replace_once(
'''        except BaseException:\n            state.effect_terminalized = True\n            raise\n        state.effect_terminalized = True\n        return result\n\n    async def _record_completed_effect(''',
'''        except BaseException as raised:\n            state.effect_terminalized = True\n            if raised is error:\n                model_error = _model_tool_error(error, call=call, tool_def=tool_def)\n                if model_error is not error:\n                    raise model_error from error\n            raise\n        state.effect_terminalized = True\n        return result\n\n    async def _record_completed_effect(''')

replace_once(
'''\ndef _bypasses_tool_error_hook(error: BaseException) -> bool:\n''',
'''\ndef _model_tool_error(\n    error: BaseException,\n    *,\n    call: ToolCallPart,\n    tool_def: ToolDefinition,\n) -> BaseException:\n    if isinstance(error, ValidationError):\n        content = RetryPromptPart.from_error(\n            error,\n            tool_name=tool_def.name,\n            tool_call_id=call.tool_call_id,\n        ).content\n        return ModelRetry(\n            _format_model_tool_error(\n                _MODEL_RETRY_PREFIX,\n                _model_tool_error_content(content, "correct the call and retry"),\n            )\n        )\n    if isinstance(error, ModelRetry):\n        return ModelRetry(_format_model_tool_error(_MODEL_RETRY_PREFIX, error.message))\n    if isinstance(error, ToolRetryError):\n        return ModelRetry(\n            _format_model_tool_error(\n                _MODEL_RETRY_PREFIX,\n                _model_tool_error_content(\n                    error.tool_retry.content,\n                    "correct the call and retry",\n                ),\n            )\n        )\n    if isinstance(error, ToolFailed):\n        return ToolFailed(_format_model_tool_error(_MODEL_FAILED_PREFIX, error.message))\n    if isinstance(error, ToolFailedError):\n        return ToolFailed(\n            _format_model_tool_error(\n                _MODEL_FAILED_PREFIX,\n                _model_tool_error_content(\n                    error.tool_failed.content,\n                    "adapt and continue",\n                ),\n            )\n        )\n    return error\n\n\ndef _model_tool_error_content(content: object, fallback: str) -> str:\n    if isinstance(content, str):\n        return content\n    try:\n        return json.dumps(\n            content,\n            ensure_ascii=False,\n            separators=(",", ":"),\n            sort_keys=True,\n        )\n    except (TypeError, ValueError):\n        return fallback\n\n\ndef _format_model_tool_error(prefix: str, message: str) -> str:\n    value = f"{prefix}: {message}"\n    if len(value) <= _MODEL_TOOL_ERROR_MAX_CHARS:\n        return value\n    tail_chars = (\n        _MODEL_TOOL_ERROR_MAX_CHARS\n        - _MODEL_TOOL_ERROR_HEAD_CHARS\n        - len(_MODEL_TOOL_ERROR_TRUNCATION_MARKER)\n    )\n    return (\n        value[:_MODEL_TOOL_ERROR_HEAD_CHARS]\n        + _MODEL_TOOL_ERROR_TRUNCATION_MARKER\n        + value[-tail_chars:]\n    )\n\n\ndef _bypasses_tool_error_hook(error: BaseException) -> bool:\n''')

path.write_text(text)
