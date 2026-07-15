# 1. 功能对齐矩阵 (Functional Parity Matrix)

> 锁定 `linktools-ai` 在激进式精简前的功能行为。每一项功能必须有一组正常测试与异常测试，精简完成后矩阵全部通过即视为功能未减少。
> 路径相对仓库根目录。

---

# 1.1 Runtime

| 功能 | 正常测试 | 异常测试 |
|---|---|---|
| Runtime.build | tests/ai/test_runtime.py::test_runtime_build_assembles_storage_compiler_runner | tests/ai/test_runtime.py::test_runtime_build_no_longer_accepts_workdir |
| Runtime.run | tests/ai/test_runtime.py::test_runtime_run_creates_session_when_none_given_and_returns_result | tests/ai/test_runtime.py::test_runtime_run_with_unknown_session_raises |
| Runtime.run_stream | tests/ai/test_runtime_stream.py::test_runtime_run_stream_yields_text_events_and_completes | tests/ai/test_runtime_stream.py::test_run_stream_rejects_swarm_spec |
| Runtime.resume | tests/ai/test_runtime_resume.py::test_resume_round_trip_pause_approve_resume_succeeds | tests/ai/test_runtime_resume.py::test_resume_non_waiting_run_raises |
| Runtime.cancel | tests/ai/test_runtime_cancel.py::test_cancel_running_run_transitions_to_cancelled | tests/ai/test_runtime_cancel.py::test_cancel_unknown_run_raises |
| Runtime.inspect | tests/ai/runtime/test_runtime_inspection.py::test_inspect_surfaces_security_degradation_as_warning | tests/ai/runtime/test_runtime_inspection.py::test_inspection_does_not_leak_handlers_or_managed_definitions |
| Runtime.aclose | tests/ai/test_custom_provider_runtime.py::test_runtime_async_context_manager_closes_mcp | tests/ai/test_custom_provider_runtime.py::test_aclose_is_idempotent |
| async with Runtime | tests/ai/test_custom_provider_runtime.py::test_runtime_async_context_manager_closes_mcp | — |

---

# 1.2 Agent

| 功能 | 正常测试 | 异常测试 |
|---|---|---|
| tools=None | tests/ai/registry/test_agent.py::test_get_applies_defaults_when_minimal | tests/ai/architecture/test_invariants.py::test_invariant_runner_requires_assembler_for_declared_tools |
| tools=() | tests/ai/test_capability_resolver.py::test_assemble_empty_tools_yields_empty_bundle | tests/ai/architecture/test_invariants.py::test_invariant_runner_empty_tools_does_not_require_assembler |
| 显式 tools | tests/ai/test_runtime.py::test_runtime_build_wires_execution_backend_for_builtin_tools | tests/ai/architecture/test_invariants.py::test_invariant_runner_requires_assembler_for_declared_tools |
| 默认文本输出 | tests/ai/test_runtime.py::test_runtime_run_creates_session_when_none_given_and_returns_result | — |
| output_schema | tests/ai/test_runtime_resume.py::test_resume_round_trip_pause_approve_resume_succeeds | tests/ai/agent/test_spec.py::test_agent_spec_construction |
| Session | tests/ai/test_runtime.py::test_runtime_run_with_explicit_session_reuses_it | tests/ai/test_runtime.py::test_runtime_run_with_unknown_session_raises |
| Cancellation | tests/ai/test_runtime_cancel.py::test_cancel_running_run_transitions_to_cancelled | tests/ai/agent/test_runner_cancel.py::test_run_cancelled_mid_lifecycle_transitions_to_cancelled |
| Checkpoint | tests/ai/agent/test_runner.py::test_run_succeeds_persists_session_run_events_and_checkpoint | tests/ai/agent/test_checkpoint_io.py::test_serialize_then_deserialize_round_trips_messages |
| Resume | tests/ai/test_runtime_resume.py::test_resume_round_trip_pause_approve_resume_succeeds | tests/ai/tool/test_executor_resume.py::test_check_allows_through_when_already_approved |
| Approval pause/resume | tests/ai/agent/test_runner_pause_nonstream.py::test_run_catches_run_paused_transitions_to_waiting_and_reraises | tests/ai/agent/test_runner_pause_atomic.py::test_sqla_pause_writes_all_three_operations_atomically_on_success |
| Middleware | tests/ai/agent/test_runner.py::test_middleware_runner_hooks_fire_in_order_on_success | tests/ai/middleware/test_pipeline.py::test_before_run_fires_in_registration_order |
| Knowledge/Retriever | tests/ai/agent/test_runner.py::test_retriever_injection_prepends_knowledge_section_to_prompt | tests/ai/knowledge/test_retriever.py::test_search_basic |
| Memory | tests/ai/agent/test_runner.py::test_memory_store_injection_prepends_memory_section_to_prompt | tests/ai/memory/test_manager.py::test_remember_then_recall_substring |

---

# 1.3 Tool

| 功能 | 正常测试 | 异常测试 |
|---|---|---|
| Builtin 文件读取 | tests/ai/test_builtin_provider.py::test_builtin_file_read_exposes_only_read_tools | tests/ai/execution/test_local_backend.py::test_write_then_read_file_roundtrip |
| Builtin 文件写入 | tests/ai/test_builtin_provider.py::test_builtin_file_write_exposes_only_write_tools | tests/ai/execution/test_local_backend.py::test_apply_patch_modifies_file |
| Builtin 终端执行 | tests/ai/test_builtin_provider.py::test_builtin_terminal_exposes_only_bash | tests/ai/execution/test_local_backend.py::test_run_bash_executes_in_runtime_dir |
| Skill Tool | tests/ai/test_skill_provider.py::test_skill_wildcard_read_skill_allowed_for_all | tests/ai/test_skill_provider.py::test_skill_single_id_only_authorized_for_that_skill |
| MCP Tool | tests/ai/test_final_hardening_combinations.py::test_mcp_tool_runs_through_managed_adapter_and_executor_to_call_tool | tests/ai/test_mcp_provider.py::test_mcp_strict_discovery_fails_closed_without_explicit_governance |
| Subagent Tool | tests/ai/test_subagent_provider.py::test_call_subagent_allowed_global | tests/ai/test_subagent_provider.py::test_call_subagent_structured_error_on_failure |
| Package Entrypoint Tool | tests/ai/package/test_package_entrypoint_execution.py::test_call_package_entrypoint_forwards_parent_identity_unmodified | tests/ai/test_package_provider.py::test_package_entrypoint_call_is_opt_in |
| Tool Exposure | tests/ai/test_tool_exposure_policy.py::test_defaults_are_conservative | tests/ai/test_security_e2e.py::test_exposure_policy_default_hides_write_and_terminal_tools |
| Tool Policy | tests/ai/tool/test_policy_adapter.py::test_known_tool_maps_risk_approval_idempotent_timeout | tests/ai/tool/test_policy_idempotency_validation.py::test_resolved_business_key_requires_key_field |
| PolicyEngine | tests/ai/policy/test_engine.py::test_deny_wins_over_allow | tests/ai/policy/test_engine.py::test_require_approval_wins_over_allow_but_not_deny |
| SecurityBaseline | tests/ai/test_security_e2e.py::test_runtime_with_default_baseline_runs | tests/ai/test_security_e2e.py::test_disabled_baseline_actually_disables_denylist_without_pause_on_approval |
| SecurityPipeline | tests/ai/security/test_composite_pipeline.py::test_later_deny_overrides_earlier_require_approval | tests/ai/security/test_composite_pipeline.py::test_after_tool_deny_result_short_circuits_like_deny |
| 参数 Schema | tests/ai/security/test_phase2a_types.py::test_adapter_modify_allows_schema_valid_payload | tests/ai/security/test_phase2a_types.py::test_adapter_modify_revalidates_against_schema_and_denies_invalid |
| 结果治理 | tests/ai/security/test_phase2a_types.py::test_adapter_after_tool_modify_result_replaces_result | tests/ai/security/test_composite_pipeline.py::test_after_tool_deny_result_short_circuits_like_deny |
| Timeout | tests/ai/tool/test_executor_timeout_retry.py::test_execute_timeout_raises_asyncio_timeout_error_when_handler_exceeds_timeout | tests/ai/agent/test_runner.py::test_run_model_timeout_transitions_run_to_failed |
| Retry | tests/ai/tool/test_executor_timeout_retry.py::test_execute_max_retries_succeeds_after_failures | tests/ai/tool/test_executor_timeout_retry.py::test_execute_does_not_retry_permanent_error |
| Approval | tests/ai/tool/test_executor_approval.py::test_check_with_approval_store_persists_pending_request_and_raises | tests/ai/tool/test_executor_approval.py::test_check_with_event_store_also_emits_approval_requested_event |
| EXACT_CALL 幂等 | tests/ai/tool/test_executor_idempotency.py::test_same_idempotency_key_calls_handler_once_and_returns_cached_result | tests/ai/tool/test_policy_idempotency_validation.py::test_resolved_idempotent_false_with_explicit_exact_call_is_rejected |
| BUSINESS_KEY 幂等 | tests/ai/test_final_hardening.py::test_business_key_requires_trusted_configured_field | tests/ai/tool/test_policy_idempotency_validation.py::test_resolved_business_key_requires_key_field |
| 安全审计 | tests/ai/tool/test_security_event_routing.py::test_policy_resolved_routes_to_security_channel | tests/ai/tool/test_security_event_routing.py::test_security_audit_failure_is_fail_closed_via_store |
| 普通观测 | tests/ai/agent/test_runner_observability.py::test_observability_records_run_and_model_spans_with_metrics_on_success | tests/ai/tool/test_security_event_routing.py::test_lifecycle_events_route_to_observability_channel |

---

# 1.4 Capability

| 功能 | 正常测试 | 异常测试 |
|---|---|---|
| builtin | tests/ai/test_capability_resolver.py::test_assemble_builtin_file_only | tests/ai/test_builtin_provider.py::test_builtin_file_exposes_only_file_tools |
| skill | tests/ai/test_capability_resolver.py::test_assemble_kindname_string_resolves | tests/ai/test_skill_provider.py::test_skill_single_id_only_authorized_for_that_skill |
| mcp | tests/ai/test_capability_resolver.py::test_assemble_kindname_string_resolves | tests/ai/test_mcp_provider.py::test_mcp_strict_discovery_fails_closed_without_explicit_governance |
| subagent | tests/ai/test_subagent_provider.py::test_subagent_wildcard_authorizes_all | tests/ai/test_subagent_provider.py::test_call_subagent_structured_error_on_failure |
| package | tests/ai/test_package_provider.py::test_package_resource_exposes_read_tools | tests/ai/test_package_provider.py::test_package_kind_is_prompt_catalog_only |
| 自定义 Provider kind | tests/ai/test_capability_resolver.py::test_register_accepts_a_new_kind | tests/ai/test_capability_resolver.py::test_register_rejects_duplicate_kind |
| Prompt Section | tests/ai/test_capability_resolver.py::test_prompt_sections_merged_in_stable_order | tests/ai/test_capability_resolver.py::test_declared_pipelines_are_merged_into_bundle |
| Exposure 过滤 | tests/ai/test_tool_exposure_policy.py::test_non_discovery_non_mutating_tool_always_exposed | tests/ai/test_tool_exposure_policy.py::test_mutating_mcp_tool_hidden_by_default_policy |
| 工具名冲突检测 | tests/ai/test_capability_resolver.py::test_cross_capability_tool_name_conflict_detected | tests/ai/test_capability_resolver.py::test_duplicate_ref_raises_conflict |
| Inspection | tests/ai/runtime/test_runtime_inspection.py::test_inspection_does_not_leak_handlers_or_managed_definitions | tests/ai/test_closure_contracts.py::test_inspect_returns_immutable_capability_inspection |
| Provider 替换 | tests/ai/test_capability_resolver.py::test_replace_intentionally_overrides_existing_kind | tests/ai/test_capability_resolver.py::test_register_rejects_duplicate_kind |

---

# 1.5 MCP

| 功能 | 正常测试 | 异常测试 |
|---|---|---|
| strict discovery | tests/ai/test_mcp_provider.py::test_mcp_strict_discovery_fails_closed_without_explicit_governance | tests/ai/test_mcp_provider.py::test_mcp_strict_discovery_fails_closed_without_explicit_governance |
| best-effort discovery | tests/ai/test_mcp_provider.py::test_mcp_best_effort_discovery_mode_opts_out_of_fail_closed | tests/ai/test_final_hardening.py::test_mcp_discovery_uses_list_tools_not_contextual_get_tools |
| 零工具服务 | tests/ai/test_mcp_provider.py::test_mcp_no_connection_manager_yields_empty | — |
| prefix | tests/ai/test_mcp_provider.py::test_final_tool_name_defaults_to_server_prefix | tests/ai/mcp/test_mcp_governance.py::test_tool_prefix_default_avoids_conflict |
| 自定义 prefix | tests/ai/test_mcp_provider.py::test_final_tool_name_custom_prefix | — |
| 无 prefix 冲突检测 | tests/ai/mcp/test_mcp_governance.py::test_cross_server_conflict_detected | tests/ai/test_mcp_provider.py::test_detect_conflicts_raises_on_duplicate_final_name |
| ConnectionRef | tests/ai/test_custom_provider_runtime.py::test_mcp_tool_runs_through_runtime_to_connection_manager | — |
| 配置 fingerprint | tests/ai/test_mcp_provider.py::test_connection_manager_cache_keyed_on_config_fingerprint | tests/ai/test_mcp_provider.py::test_connection_manager_closes_toolsets |
| env/header 变化 | tests/ai/test_mcp_provider.py::test_parse_stdio_spec_structured_fields | tests/ai/test_mcp_provider.py::test_connection_manager_cache_keyed_on_config_fingerprint |
| description | tests/ai/test_managed_definition_description.py::test_mcp_description_flows_into_managed_definition | — |
| readOnlyHint | tests/ai/test_managed_definition_description.py::test_mcp_read_only_hint_marks_descriptor_non_mutating | tests/ai/test_managed_definition_description.py::test_convert_tool_info_reads_read_only_hint_false |
| exposed_name 治理 | tests/ai/test_mcp_provider.py::test_mcp_disabled_tools_removed_from_toolset_surface | tests/ai/test_mcp_provider.py::test_mcp_multi_server_disabled_tools_filter_independent_per_server |
| raw_name 实际调用 | tests/ai/test_closure_contracts.py::test_mcp_descriptor_carries_raw_name_for_audit | tests/ai/test_final_hardening_combinations.py::test_mcp_tool_runs_through_managed_adapter_and_executor_to_call_tool |
| 完整 Runtime call | tests/ai/test_final_hardening_combinations.py::test_mcp_tool_runs_through_managed_adapter_and_executor_to_call_tool | tests/ai/test_custom_provider_runtime.py::test_mcp_tool_runs_through_runtime_to_connection_manager |

---

# 1.6 Swarm / Subagent

| 功能 | 正常测试 | 异常测试 |
|---|---|---|
| SwarmSpec | tests/ai/swarm/test_spec.py::test_swarm_spec_constructs_with_required_fields | tests/ai/swarm/test_spec.py::test_swarm_spec_is_frozen |
| Agent 映射 | tests/ai/swarm/test_runner.py::test_run_parallel_fan_out_aggregates_and_marks_succeeded | — |
| Subagent depth | tests/ai/test_subagent_provider.py::test_call_subagent_depth_exceeded | tests/ai/swarm/test_strategy.py::test_run_task_raises_when_depth_exceeds_max_depth |
| root_run_id | tests/ai/swarm/test_runner.py::test_run_parallel_fan_out_aggregates_and_marks_succeeded | — |
| parent_run_id | tests/ai/swarm/test_strategy.py::test_coordinator_delegation_runs_two_workers_and_aggregates | — |
| session_id | tests/ai/swarm/test_runner.py::test_run_parallel_fan_out_aggregates_and_marks_succeeded | tests/ai/test_subagent_provider.py::test_call_subagent_allowed_global |
| timeout | tests/ai/swarm/test_runner.py::test_run_timeout_transitions_driving_run_and_swarm_to_failed | tests/ai/swarm/test_runner.py::test_run_max_total_tokens_exceeded_raises_and_marks_failed |
| 错误传播 | tests/ai/swarm/test_strategy.py::test_run_task_complete_task_not_found_propagates | tests/ai/swarm/test_strategy.py::test_run_task_complete_task_storage_error_propagates |
| Package scope | tests/ai/test_package_scoped_subagent.py::test_scoped_subagent_resolves_via_entrypoint | tests/ai/test_package_scoped_subagent.py::test_scoped_subagent_undeclared_package_rejected |

---

# 1.7 Storage

| 功能 | 正常测试 | 异常测试 |
|---|---|---|
| File Backend | tests/ai/storage/test_facade.py::test_file_storage_runs_end_to_end | tests/ai/storage/test_facade.py::test_file_storage_constructs_full_facade_with_file_capabilities |
| SQLAlchemy Backend | tests/ai/storage/test_facade.py::test_sqlalchemy_storage_runs_end_to_end | tests/ai/storage/test_facade.py::test_sqlalchemy_storage_constructs_full_facade_with_sql_capabilities |
| SQLAlchemy 可选依赖 | tests/ai/test_optional_imports.py::test_core_imports_without_sqlalchemy | tests/ai/test_residual_cleanup.py::test_import_linktools_ai_without_sqlalchemy |
| 懒加载 | tests/ai/test_public_api.py::test_sqlalchemy_storage_accessible_lazily_from_storage_package | tests/ai/test_public_api.py::test_root_does_not_re_export_sqlalchemy_storage |
| 不同 Store 不同 Backend | tests/ai/storage/contract/test_run_store_contract.py | tests/ai/storage/contract/test_event_store_contract.py |
| Run Store | tests/ai/storage/contract/test_run_store_contract.py::test_create_then_get_roundtrip | tests/ai/storage/contract/test_run_store_contract.py::test_concurrent_transitions_with_same_expected_version_only_one_succeeds |
| Session Store | tests/ai/storage/contract/test_session_store_contract.py::test_create_then_get_roundtrip | tests/ai/storage/contract/test_session_store_contract.py::test_concurrent_append_messages_never_assigns_duplicate_sequence |
| Event Store | tests/ai/storage/contract/test_event_store_contract.py::test_append_then_list_roundtrip | tests/ai/storage/contract/test_event_store_contract.py::test_concurrent_appends_get_distinct_sequences |
| Checkpoint Store | tests/ai/storage/contract/test_checkpoint_store_contract.py::test_save_then_get_roundtrip | tests/ai/storage/contract/test_checkpoint_store_contract.py::test_latest_returns_highest_sequence |
| Approval Store | tests/ai/storage/contract/test_approval_store_contract.py::test_create_then_get_roundtrip | tests/ai/storage/contract/test_approval_store_contract.py::test_approve_transitions_to_approved |
| Idempotency Store | tests/ai/storage/contract/test_idempotency_store_contract.py::test_reserve_complete_get_roundtrip | tests/ai/storage/contract/test_idempotency_store_contract.py::test_reserve_same_scope_key_with_different_hash_raises_conflict |
| Memory Store | tests/ai/storage/contract/test_memory_store_contract.py::test_remember_then_get_roundtrip | tests/ai/storage/contract/test_memory_store_contract.py::test_search_filters_substring_and_limit |
| Swarm Store | tests/ai/storage/contract/test_swarm_store_contract.py::test_create_run_then_get_run_roundtrip | tests/ai/storage/contract/test_swarm_store_contract.py::test_claim_task_marks_claimed_and_assigns_agent |
| Resource Store | tests/ai/storage/resource/test_store.py::test_primary_only_put_get_roundtrip | tests/ai/storage/resource/test_store.py::test_overlay_fallback_when_primary_missing |
| Event round-trip | tests/ai/architecture/test_event_round_trip.py::test_all_event_payloads_round_trip_through_file_store | tests/ai/architecture/test_event_round_trip.py::test_every_standard_payload_is_covered |

---

# 1.8 CLI

| 功能 | 正常测试 | 异常测试 |
|---|---|---|
| 轻量 CLI | tests/ai/commands/test_support.py::TestResolveModelConfig::test_flags_take_precedence_over_env | tests/ai/commands/test_support.py::TestValidateSessionId::test_normal_session_id_passes_through_unchanged |
| 验证最小 Runtime 链路 | tests/ai/commands/test_init_and_assembly.py::TestBuildCliRuntime::test_build_cli_runtime_wires_registries | — |

---

# 2. 已知覆盖缺口（精简前需补齐或确认）

```text
1. CLI「验证最小 Runtime 链路」的构建（build_cli_runtime 装配真实 Runtime）由
   tests/ai/commands/test_init_and_assembly.py::TestBuildCliRuntime 覆盖；
   端到端 Runtime.build -> run 的完整链路仍建议在后续补充。
2. EXACT_CALL 幂等的具名 happy-path 用例缺失，当前由 test_executor_idempotency.py 的默认缓存行为间接覆盖。
3. MCP env/header 单字段切换触发 fingerprint 变化的显式用例缺失（当前由 config-fingerprint 聚合用例覆盖）。
```
