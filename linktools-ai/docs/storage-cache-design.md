# Storage 缓存与缓冲写入统一设计

状态：72734d3已发布范围设计复审通过；本次修订已纳入两种缓冲读取可见性。新增可读封装的公共形状仍待收敛，新增范围未实现，也不以旧稿复审结论代替新增范围验证。
日期：2026-10-15
范围：Issue #190 / PR #191

## 1. 目标与边界

以现有 storage 能力为基础减少重复读取、重复回源和可合并的写入。公共接口保持少量明确概念，业务调用不散落缓存 key、TTL、失效与刷新逻辑。允许对尚未发布的接口作必要破坏性调整，不保留旧别名。用户明确不要求兼容旧持久数据；不设计旧数据迁移或旧格式读取分支，也不因此授权删除实际环境数据。最新代码的功能、现有调用能力与扩展语义必须完整保留，必要接口调整同步所有调用方。

Redis 连接、配置及适配器归宿主。框架不依赖 Redis，不添加 Redis 专属参数。内存、文件、宿主共享缓存按真实复用收益组合，不强制三层。缓存不替代授权、权威存在性、保留期、事务与恢复记录。

缓冲写入与读缓存有不同职责：前者尚未获得 sink 的持久确认，不能像命中的副本一样任意逐出；后者可以丢弃并回源。本稿内存 WriteBuffer 只用于业务明确接受延迟及异常丢失风险的数据；两种读取可见性均可使用它，不因只读已提交数据就改变交付保证。

## 2. 语义与所有权

| 概念 | 语义与权威 | 保留的接口族 |
|---|---|---|
| 逻辑条目 | 可更新值、entry 修订、条件写入、DELETE/RESET | StorageReader/Writer、批量读写协议 |
| 单一源 | 条目内容及 metadata/revision 的来源 | backend |
| 组合视图 | 优先级层、有效条目与 metadata 派生视图 | StorageOverlay |
| 字节对象 | 可引用的不可变字节成员及完整性 | ObjectStore/ObjectRef/ObjectStat |
| 字节缓存后端 | 可丢弃的字节副本，无源存在性/授权权威 | CacheBackend |
| 领域内容缓存 | 持有借用的字节后端，定义领域 key 与编码/解码 | ContentCache；AssetContentCache |
| 持久内容描述 | inline 内容或外部 ObjectRef | StoredPayload |
| 事务状态 | record/alias/sequence/fact/operation 与 CAS | StateStore |
| 修订提示 | 加速 metadata 新鲜度探测；不是提交凭据 | RevisionSource |
| 写缓冲 | 被接纳但尚未获得 sink 确认的批写项 | 单独的条目有界缓冲职责，接口见第8节 |

ObjectStore 和 CacheBackend 不能为了词形一致合并。get 返回可缺失的完整值，stat 返回描述，open 返回字节流，read_object 完整物化并校验引用；这些不同名称有实际语义依据。

## 3. 全局命名与合同收敛

### 已有充分依据的改动

| 当前 | 推荐 | 原因 |
|---|---|---|
| StorageOverlay 缓存注入 | content_cache=AssetContentCache(cache_backend) | 单一领域对象包含字节后端与 key/codec |
| RuntimeStorage 字节缓存注入 | cache_backend | Runtime 已有 ObjectRef/domain/tenant/source 规则 |
| InMemoryCacheBackend.max_size | max_entries | 单位是条目数，区别 max_bytes |
| preload_concurrency | origin_read_concurrency | 普通读取 cache miss 同样使用 |
| cache_concurrency | cache_read_concurrency | 实际限制 cache probe |
| StorageOverlay.refresh / invalidate | refresh_metadata / invalidate_metadata | 不清空内容副本，只操作 metadata |
| RevisionSource.revision_bumped | notify_revision | 可能纠正错误提示，不能承诺单调递增 |
| 具体类 StorageRevisionSource | BackendRevisionSource | 实际把 backend.head_revision 适配为 RevisionSource |
| InitializableStorage | StorageLifecycle | 契约同时含 initialize 与 close |

缓存接入使用两层职责：CacheBackend 仅保存字节，具体实现为 InMemoryCacheBackend、FilesystemCacheBackend、TieredCacheBackend；ContentCache 持有 backend 并定义 cache_key/to_cache/from_cache。AssetContentCache(backend) 直接实现该领域职责，StorageOverlay 只有 content_cache 一个缓存参数，不再另传适配器。RuntimeStorage 使用 cache_backend，沿用自身引用、租户与源存在性校验，不新增 Runtime 适配器或 options/binding 封装。两者借用后端而不关闭它；单飞保留 value 与 encoded bytes，每个 reader 独立解码可变值。StorageId/StoragePath 推荐 PathSegment/RelativePath，准确表达文件路径约束；两者没有生产调用者，不添加 Storage 长前缀。不机械给所有 DTO 添加前缀，不增加总 manager/spec/registry。

修订作用域通过类型与所属对象明确：entry_revision 是单条目历史修订；backend.head_revision 是单源 head；Overlay.current_revision 是组合视图 head；写结果 store_revision 仍指实际 writer head。保留这些现有名称并补足合同，不为前缀整齐增加 effective_revision/backend_revision 两套平行术语。RESET 撤销写层覆盖、DELETE 产生删除状态，不能合并为同一动作。

### 批量读取合同

BatchStorageReader 与四个内置 Asset backend 的 get_many(keys) 返回 Mapping[key, value]，只包含存在且可见的条目；重复 key 合并，缺失 key 不进入结果。StorageOverlay、AssetStore、快照和能力捕获的 AssetStoreReader 返回 tuple[value | None, ...]，按输入顺序保留重复 key 与缺失位置。两层均通过 await 完成批读后返回，保留 backend 批量查询及 Overlay 的按 key 合并，不提供流式前缀或额外 iter_many 接口。ExecutionRepository 的领域 Mapping 合同保持独立。

### 持久字段与当前代码一致性

AssetInfo、AssetVersionRef 和 CapabilityLoadEntry 的 digest 字段表示内容字节的 SHA-256，采用 64 位小写十六进制。当前 SQL JSON payload、文件 metadata/history、batch receipt、快照及 Runtime 内嵌引用统一使用 digest；HTTP ETag 保持独立的传输语义。

StoredRecord.storage_revision 与 Observed.storage_revision 是物理 CAS token；RecordReplacement 和 StateTransaction 使用 expected_storage_revision 作为比较条件。guard/lease 操作也会递增 storage_revision，但不改变领域数据及其 revision。业务 revision/expected_revision 保留各自语义，version 仍用于 wire/schema 格式。当前 ai_state_records 列和文件 record codec 均使用 storage_revision，不保留旧字段别名或旧数据读取、迁移路径。Asset digest 改名仍须同步当前 writers/readers/快照引用及摘要生成，不能用旧 ObjectRef 指向改过的字节。旧持久数据兼容不在范围，但不得自动清库。

## 4. 场景矩阵

| 场景 | 推荐方式 | 必须守住的边界 |
|---|---|---|
| 版本化资产内容 | 有界内存，按需要加文件/宿主共享副本 | key 绑定逻辑条目、entry revision 与 digest；读前由所属源的 metadata 和调用方权限决定可见性 |
| 资产 metadata | 进程快照 + RevisionSource 提示 | metadata刷新遵循来源策略；显式权威读取由backend所有者负责，不由提示代替 |
| 持久 history/result 大正文 | 独立预算内容缓存，复用 Runtime object router | DURABLE 才接入；每次仍守存在性、授权、保留期 |
| 短期配置/派生参数 | 复用宿主已有 TTL 与回源合并 | 失效由业务版本/刷新契约决定，不统一猜 TTL |
| 实时片段与 UI projection | 已有进程内订阅/有界状态 | 不是写库后再轮询；重连仍尊重 cursor/高水位 |
| session/graph/权限/控制 | 保留权威状态路径 | 不把租约、fence、授权、幂等当普通副本 |
| 可观测性指标/诊断 | 评估有界批写 | 先明确哪些允许延迟/丢失；记账和审计不能默认可丢 |
| 快照/恢复/冻结资产 | 各自固定读取源 | 不以最新在线缓存替代冻结内容 |

资产缓存允许同一摘要的字节跨来源复用；现有 AssetContentCache 不提供租户或物理 backend 的 key 隔离。需要独立隔离的宿主使用适配器 namespace 或独立缓存实例，不增加框架参数。Runtime 对象缓存采用另一所有者合同：key 包含 namespace、tenant、domain 与完整 ObjectRef；store_id 不是全局物理源标识，命中仍验证当前实际源中的对象存在性。

## 5. 最小接入方式

已有：CacheBackend、内存/文件/Tiered实现、ObjectStore、StorageOverlay。PR191已实现但尚未整体交付：RuntimeStorage.cache_backend及内部读取路由。Storage 与 Asset 批读沿用第3节的分层合同；其余拟调整命名见第3节，写缓冲公开形状在第8节。

已有 CacheBackend 继续仅含 get/put/delete/contains_many；TieredCacheBackend 可以嵌套复用，不引入层配置 DSL。

### 5.1 资产读写：拟调整接口的完整示例

下面代码的 content_cache 和 digest 已落实；BackendRevisionSource、max_entries 仍为第3节拟实施名称，其相应导出与实现同步落地后示例可直接运行。示例仅创建进程内资产，不读取、覆盖或删除实际环境数据。

    import asyncio
    import hashlib

    from linktools.ai.asset import (
        AssetContentCache, AssetKey, AssetStore, InMemoryAssetBackend,
    )
    from linktools.ai.storage import (
        BackendRevisionSource, InMemoryCacheBackend, StorageOverlay,
    )

    async def asset_example() -> None:
        backend = InMemoryAssetBackend()
        revision_source = BackendRevisionSource(backend)
        store = AssetStore(StorageOverlay(
            primary=backend,
            writer=backend,
            revision_source=revision_source,
            content_cache=AssetContentCache(InMemoryCacheBackend(
                max_entries=128, max_bytes=1024 * 1024,
            )),
        ))
        try:
            await store.initialize()
            key = AssetKey("text", "greeting")
            missing = AssetKey("text", "missing")
            content = b"hello"
            info = await store.put(key, content)
            assert info.digest == hashlib.sha256(content).hexdigest()
            assert await store.get_many((key, missing, key)) == (
                content, None, content,
            )
            assert await store.get(key) == content
        finally:
            await store.close()

    if __name__ == "__main__":
        asyncio.run(asset_example())

普通 store.put 已在提交后由 Overlay 调用 revision_source.notify_revision(result.store_revision)，业务不重复通知。BackendRevisionSource 直接观察 backend.head_revision，其 notify_revision 无需保存副本；宿主若提供共享提示适配器，仍实现 RevisionSource 的 head_revision/notify_revision，只有适配器负责提示存储。直接绕过 Overlay 的外部写入需遵守该来源的通知/刷新合同，不能仅写提示而不提交资产。

### 5.2 已有 Runtime 历史：现有业务 API 与完整生命周期

此例使用 PR191 已有接口。runtime_root 必须是已有 RuntimeStorage.from_root 使用的同一目录，namespace、principal 及 authorization 必须对应该数据的原租户和授权策略。session_id 与 execution_id 指向可读取的既有会话和已成功完成、结果及持久 transcript 仍可读取的 Agent 执行；仅有结果而未保存 transcript 的执行不满足此例前提。本例不会创建业务记录。cache_root 是独立、专用于可丢弃副本的目录。

    from pathlib import Path

    from linktools.ai.core import AuthorizationPolicy, Principal
    from linktools.ai.runtime import RuntimeHistory, RuntimeStorage
    from linktools.ai.storage import (
        CacheBackend, FilesystemCacheBackend,
        InMemoryCacheBackend, TieredCacheBackend,
    )

    async def read_existing_history(
        runtime_root: Path,
        cache_root: Path,
        namespace: str,
        principal: Principal,
        authorization: AuthorizationPolicy,
        session_id: str,
        execution_id: str,
        *,
        shared_cache: CacheBackend | None = None,
    ) -> None:
        lower_cache: CacheBackend = FilesystemCacheBackend(
            cache_root, max_bytes=64 * 1024 * 1024,
        )
        if shared_cache is not None:
            lower_cache = TieredCacheBackend(lower_cache, shared_cache)
        cache_backend = TieredCacheBackend(
            InMemoryCacheBackend(max_bytes=16 * 1024 * 1024),
            lower_cache,
        )
        storage = RuntimeStorage.from_root(
            runtime_root, cache_backend=cache_backend,
        )
        async with RuntimeHistory.open(
            namespace,
            storage=storage,
            tenant_id=principal.tenant_id,
            authorization=authorization,
        ) as history:
            print(await history.inspect_session(session_id, principal=principal))
            timeline = await history.session_timeline(
                session_id, principal=principal, limit=20,
            )
            transcript = await history.transcript(
                execution_id, principal=principal,
                include_content=True, limit=20,
            )
            result = await history.result(execution_id, principal=principal)
            print(timeline.items, transcript.items, result.output)

RuntimeHistory.open 在进入时按 read_only=True 初始化所传 storage，在退出时关闭它，包括业务读取异常的退出路径。此例应传入新构造的 storage，不借用正在运行的 Runtime 所拥有的实例。示例仅读首个 timeline/transcript 页面；完整分页继续使用返回的 next_cursor 与相同查询参数。构造函数不会替调用者找已有会话，目录和身份参数需由宿主提供。

有 shared_cache 时依次检查内存、文件、宿主共享层，命中向前层提升；put 向配置的层写入。只需要内存或两层时直接删减组合，避免无收益副本与写放大。示例预算用于展示，不是固定最佳值；Runtime 和资产使用独立预算，避免大正文挤出资产缓存。

宿主共享缓存适配器实现同一 CacheBackend；连接配置与关闭仍由宿主资源所有者负责，RuntimeHistory 退出不关闭它。适配器须限制网络操作耗时；低层 TTL 不会自动使已提升到内存/文件层的副本过期，业务失效仍由拥有者合同决定。

RuntimeStorage.read_object(domain, reference) 是 storage-bound SDK 读出口，供内部组装 history/transcript/result 复用；普通业务不主动调用它。它与通用 storage.read_object(store, key, ...) 的完整物化/校验工具具有不同所属对象和参数。独立冷 miss 创建新 flight，直接读取源而不先 stat；后到冷 miss 若加入已有 flight，须另查权威源，避免借用源已删除的旧读取；热 hit 校验缓存字节并检查权威对象存在性。查询数按这三类分别计量。禁用缓存和非 DURABLE 保留原有取消与源读取行为。

## 6. 并发、一致性与生命周期

- 同一 owning loader 内的冷 miss 共享已有 flight；共享 CacheBackend 的不同 Runtime 实例不自动获得分布式 single-flight
- completed flight 不作为新请求的缓存；权威 revision 仅允许源 I/O 开始前同批读共享，后到读必须新观察
- 已知更新的 generation/修订不能被旧刷新覆盖；opaque SDK revision 不擅自排序，宿主可按自身版本合同处理
- revision通知必须绑定实际writer的来源：现有Overlay允许非primary writer，而唯一revision_source在primary view。不能把其它层writer的head发布给primary提示源；实现按writer对应view选择source，无source则不误发，不禁止合法非primary writer，也不为此新增参数。该现存接线问题纳入实施必须修复项
- 取消一个共享读等待者不取消 producer；禁用缓存的原请求取消语义保持
- close 先封闭新工作，完成已接纳的必要工作，再按所有权关闭；外部 cache/连接不由 Runtime 接管
- archive/projection preflight 可以合法读取对象，必须先完成，再封闭 object router 并 drain
- 字节缓存异常/损坏可回源；RevisionSource.head_revision异常按现有契约传播，适配器只有明确返回None时才进入源回退，不新增吞异常降级。refresh_metadata仍遵循来源策略，不是强制绕过提示的新接口；源失败、删除、授权失败不能伪装成缓存成功
- transient/volatile 不因文件或共享缓存延长生命周期；缓存逐出不是物理安全删除保证

提交结果未知与提示通知不构成同一事务：请求取消可能发生在数据库已提交而提示尚未更新的窗口。本方案保持已有请求取消/提交结果未知合同，不把关闭修复变成无条件屏蔽取消；提示仅承诺其来源定义的最终刷新行为，不承诺固定提交后新鲜度上界。需要强一致的调用继续权威读取/readback。若未来要求提示与提交原子联动，应单独设计持久通知机制，而不在缓存层偷偷加入可靠队列。

## 7. 共享文件缓存：纯文件与共享真实用量

同一规范化 root 的所有进程通过现有 filelock 协调，对真实共享用量计量；不采用进程隔离、SQLite、新数据库依赖或进程注册表。

max_bytes 只保留在各实例内存中。每次成功写入的put持共享锁，按本实例额度对root真实用量逐出：小额度实例可压低用量，大额度实例以后仍可写高；小实例不运行时不会清理。这是用户明确接受的语义，不是永久共享最小硬上限。不持久quota，也不新增resize接口。

逐次全目录扫描原型虽通过50项定向测试，但16384项暖 put 约97.20ms，性能门未通过，不发布。目录时间戳不是可靠变更序列；锁保证互斥，却不能使旧的进程本地容量索引自动正确。

最小实现方向是一个共享小计量记录，仅包含总charge和dirty状态。每次mutation持锁读取记录，先原子标dirty，再修改内容与逐出，最后原子写入clean计量。正常只stat目标及实际逐出候选，本地索引仅排热度、不信旧size；候选不足时容量压力触发扫描。每实例首次mutation无条件锁内全扫描校准，记录缺失、损坏、dirty时也扫描恢复。失败不能写假clean。没有完整共享entry索引、generation或后台服务。

取消等待mutation worker结算后才释放锁。该缓存允许丢失，不新增数据库式断电持久性保证；机器重启后首次mutation校准，不能称只读打开即完成修复。保留不受管理文件，不跟随symlink做清理，不支持旧writer绕过协议混跑。

预算继续是已提交payload逻辑计费，空内容至少1字节，并非OS实际块分配或临时文件峰值。实现正在定向验证，尚未交付；SQLite和持久最小quota均已否决，不是备选待确认项。

## 8. 缓冲批量写入

首批仅复用 Runtime 已有且明确 best-effort 的 automatic observations，不改变审计、恢复或控制数据的持久保证。以现有 Runtime _MetricBuffer 提炼通用调度与结算；Runtime 保留 Observation 关联字段增强，sink 继续使用 Metrics.record_observations → MetricStore.put_observations。自动观测已有批写机制；当前窗口调整为满128条或首条接纳后3秒，先到即提交。后续入队不重置窗口，显式 flush 与正常 close 会主动唤醒部分批次。3秒是正常调度等待上限，不包含 sink 执行耗时；进程崩溃仍可能丢失未结算的内存观测。此调整不改变显式 Metrics.record 或关键业务事务。

### 唯一公共形状

    from linktools.ai.storage import WriteBuffer

    buffer = WriteBuffer(
        metrics.record_observations,
        max_entries=1024,
        batch_size=128,
        flush_interval_seconds=0.1,
    )
    try:
        accepted = buffer.try_submit(observation)  # 仅表示接纳
        if not accepted:
            report_metric_delivery_degraded(buffer.status())
        result = await buffer.flush(timeout_seconds=5.0)
        if result.pending or result.unconfirmed or result.dropped:
            report_metric_delivery_degraded(result)
        status = buffer.status()  # 全生命周期累计，可能包含后来接纳的条目
    finally:
        await buffer.close()

WriteBuffer[Item] 接受一个异步 write_batch(tuple[Item, ...]) 回调。公共操作仅 try_submit(item) -> bool、status() -> WriteBufferStatus、flush(*, timeout_seconds=5.0) -> WriteBufferFlushResult、close() -> None。max_entries、batch_size 必须为正整数，batch_size <= max_entries；flush_interval_seconds 必须为有限正数，flush 的 timeout_seconds 必须为有限非负数，bool 不视为数字参数。

结果模型只用 WriteBufferStatus 与 WriteBufferFlushResult，直接替换现有 MetricBufferStatus 与 MetricFlushResult；不保留兼容别名、平行结果类或另设config类，RuntimeMetrics.status/flush 同步返回通用模型。WriteBufferStatus字段含义如下：

| 字段 | 含义 |
|---|---|
| enabled | 是否配置缓冲；实例为True，未配置指标的Runtime返回False |
| accepting | 当前是否继续接纳 |
| max_entries | 全部未结算条目的上限，包含队列和正在组批/写入的条目 |
| accepted / rejected | 生命周期累计接纳数 / 未接纳数，rejected不包含于accepted |
| confirmed | sink回调成功返回的条目数；持久含义由sink的成功合同定义 |
| unconfirmed | 已进入sink回调但结果未确认的条目数；可能已提交，不能称为已丢失 |
| dropped | 已接纳、明确未调用sink便终止的条目数 |
| pending | 已接纳而尚未结算的条目数，包含正在执行的回调 |
| queue_size / high_watermark | 当前排队条目数 / pending的历史峰值 |
| write_failures / last_failure_code | sink失败批次数 / 最近失败的安全分类，不保存任意异常对象或敏感文本 |

WriteBufferStatus计数都是单个buffer全生命周期累计或当前状态；始终满足 accepted = confirmed + unconfirmed + dropped + pending。WriteBufferFlushResult只含target、confirmed、unconfirmed、dropped、pending五个字段，只描述捕获的已接纳前缀，满足 target = confirmed + unconfirmed + dropped + pending。不添加容易误读为持久成功的completed/success字段。未配置Runtime指标时返回enabled=False的零状态，flush返回target=0的零前缀结果。

### 接纳、组批与sink合同

- max_entries 是条目数，不是任意 Item 的内存字节预算。调用者须限制单项大小，并在成功接纳后移交只读使用责任；buffer 不深拷贝、不编码，也不容许调用者或sink随后修改已接纳项。Runtime的Observation已限制字段数量/长度并冻结映射
- 满载时拒绝新项、增加rejected并返回False；不逐出已接纳项，不阻塞生产者。接纳只需当前事件循环内的同步步骤，不持久化、不承诺进程崩溃恢复
- 单一writer按接纳顺序调用sink，没有并行批次、重排或自动重试。无空转轮询；首项唤醒writer，按批量或最长组批间隔提交。间隔仅限制组批等待，不是排队/写库端到端SLA
- 回调成功表示该整个批次达到sink声明的提交标准。异常、超时或取消后的不确定结果，整批记unconfirmed；callback即使可能部分成功，通用buffer也不猜测部分提交。只有明确从未调用callback的条目可记dropped
- Metrics sink保留已有稳定Observation ID、256条/16MiB批上限、原子写及commit-unknown读回。Runtime继续用128条默认批次。其他调用者配置batch_size时必须服从自己的sink限制；按字节分割和业务校验归sink/适配器，不新增第二codec，也不允许参数绕过原校验
- 首版沿用已有5秒写超时、10秒关闭预算作为内部默认，不增加timeout/config/spec类或策略枚举。业务变换、幂等ID、事务和真实底层I/O超时由sink所有者负责

### flush是已接纳前缀屏障

flush在第一次await前捕获当前accepted序号作为target，并主动唤醒组批；只等待这个前缀结算，后来接纳的条目不会延长其完成条件。返回的五个字段都是从序号1到target的累计快照，不是自上次flush以来的增量，也不含后来接纳的条目。pending=0仅表示前缀已结算；confirmed=target才表示该前缀全部得到sink成功确认。超时返回当时的前缀快照，未结算项保持pending；取消只取消该flush等待者，不取消writer。

status()返回全生命周期累计，可能包含flush之后接纳的条目，不能用其数值替代前缀结果。flush结果在返回前已冻结，后续写入不会改变它。业务需要逐项确认、失败自动恢复或可靠队列，应继续用相应权威写API，不扩展这款best-effort buffer。

内部维护连续已结算序号、累计计数及当前flush等待者。单writer按序结算；可以在最近的活动target处切批并冻结计数，再处理后续序号；同target共享屏障，超时/取消注销其等待者。超时快照在注销前同步截取，因此后续序号不会混入。活动屏障只随尚在等待的flush调用存在，不保留逐项或历史批次ledger。

关闭预算耗尽时，先结算正在处理的批次，再按序丢弃尚未交给sink的队列尾部；遇到活动target仍冻结对应前缀。不得用“任意位置已结算的总数”冒充连续前缀，不能让后项被丢弃掩盖前项仍在执行。

### close与取消

close在第一次await前停止接纳，并创建/复用唯一owned cleanup task。重复close共用同一清理；等待者取消不取消cleanup，清理结算后再向被取消等待者传播取消。覆盖从取出首项、组批到回调的整个生命周期，避免queue为空而pending永久悬挂。

正常关闭排空所有已接纳项。达到10秒预算后停止继续调用sink，要求正在执行的回调合作取消；回调结束后将无法确认的已调用项计入unconfirmed、剩余未调用项计入dropped，再结束writer。callback正常成功返回的项仍计confirmed，不因曾请求取消便假称未确认。close返回时writer已结束且pending=0；关闭预算超限和未确认/丢弃必须通过status及限频日志可见，close正常返回不等于全部写入成功。

10秒是触发停止/取消的预算，不是对任意callback的硬截止。asyncio超时包装不能强行终止吞掉取消的回调；callback必须合作取消并自行设置真实I/O超时，且返回或传播取消前结算自己启动的写I/O，不能留下脱离所有权的后台写任务/线程。不符合此合同的callback可能使close持续等待，也无法由buffer保证隐藏I/O已经停止。此时仍在执行的项保持pending，不能虚报已终止、已丢弃或已经安全关闭，也不能先释放其借用的数据库/Redis连接。buffer不拥有sink或连接；宿主在buffer真正关闭后再关闭它们。

### 首批边界与证明

业务权威账本、终态提交、事务通知和恢复快照不自动降级；新增领域必须显式选择其可损失数据，不把已有异步通知机制一并改造。

- 当前接入只包括Runtime自动观测指标：runtime/_metrics.py:82、330；observe/_metrics.py:178；observe/_store.py:93；observe/_sql.py:400
- 执行事件已有延迟批写和强制持久边界（runtime/state/_steps.py:1990–2221）；TransientObjectStore是活跃执行的临时事实所有者，均不改成可丢写入

独审已复现现有缓冲关闭取消缺陷：writer取首项后处于组批阶段，close等待者取消导致writer取消，原finally未覆盖该阶段，最终accepted=1、pending=1而queue=0。提炼前必须修复并回归；不能将现有实现原样搬到storage就算完成。新增验证至少包含：max_entries含inflight、拒新计数、定时/主动flush、前缀后持续接纳、sink成功/抛错/不确定提交、组批中取消、写入中取消、重复close、关闭预算耗尽及合作取消边界。

## 9. 验证与阶段事实

已发布 PR191 a073a7b3：metadata/共享观察修复、Runtime DURABLE 内容缓存及生命周期接线。251项定向组合通过，独审122+9项通过，CI run38718857008九项成功。不是本稿全部需求实现。

机制实验：SQLite4MiB正文读取10次，源传输40MiB降至4MiB，但均10条SELECT。内容缓存不能声称消除逐波SQL。P0根目标仍包括逐波读消除：使用已建立的实时生命周期/订阅事件路径，不再为每波文本刷新完整模型观察树；剩余权威状态、树和执行边界读由既有观察读取优化独立归属跟进，不把正文缓存当其替代。冷/禁用/命中/缓存错误路径均需按各自实际源调用计量。

### 必须验证矩阵

- 批读：重复/缺失/空输入，所有backend与facade等价，不增SQL
- 缓存：冷/热/关闭/禁用/错误/损坏/删除/保留期/授权，查询数与传输量分开
- revision：同批/后到/刷新失败/旧刷新覆盖/写通知/取消/关闭/多实例，以及非primary writer不污染primary提示源
- 文件：双实例顺序反例、跨进程并发、替换/删除、扫描/逐出失败、取消、崩溃、旧文件复用、空字节计费
- 生命周期：初始化与close、preflight读写、重复close、等待者取消、借用资源不误关
- 写缓冲：容量、定时与主动flush、并发接纳、取消、异常提交结果、重试幂等、close排空和超时/丢失边界
- 同步改造/当前格式一致性：所有导出/调用/测试/文档同步；当前JSON/SQL/快照/receipt生成和读取双端验证，不实现旧数据迁移

## 10. 实施边界及交付标准

本轮首批仅接入已有允许损失的 Runtime 自动观测契约，采用明确拒新与失败不自动重试。框架能力供其他领域显式 opt-in；阶段统计等新增业务接入需各自确认，不阻塞本轮框架设计。

旧数据范围已明确：不保留旧格式兼容或迁移工具，不执行实际环境数据删除。当前代码各 backend、调用方、快照生成与读取必须同步，不能留下只改一端的状态。

其他常规命名与组装选择由设计审查负责，不逐参数要求使用方决定。第1至12节边界已进行独立复审；第13节新增读取可见性范围单独标明尚未收敛部分。实现后须按矩阵验证。不得把静态审查或局部CI通过扩张成全部生产场景无问题保证。

## 11. 可核对来源

- 需求范围：https://github.com/linktools-toolkit/linktools/issues/190
- 已发布增量：https://github.com/linktools-toolkit/linktools/pull/191
- 已核实 CI：https://github.com/linktools-toolkit/linktools/actions/runs/38718857008
- 已发布代码基线：a073a7b3dc0d35b5c9a0f4d4c60c44752b1d6252
- 核验文件：storage/_contracts.py、_composition.py、_cache.py、_revision.py、_lock.py；asset 各 backend；runtime/state/_root.py、_object_router.py；runtime/_metrics.py；observe/_metrics.py、_store.py、_sql.py

所有“拟调整”需在实施后的精确 head 重跑相应验证；本稿不把尚未落地的文件并发修复、命名同步与 WriteBuffer 提炼记作已完成。

## 12. 已发布范围设计复审结论

适用规则：根目录及 linktools-ai/AGENTS.md，尤其公共接口、单一所有权、revision/version 词汇、当前格式双端一致及 filelock 协调规则。审查覆盖本稿职责、名称、批读、多级缓存、修订提示、共享文件预算、写缓冲前缀与生命周期合同。

本轮已订正冷读后到借用者stat、来源提示绑定、初始化关闭、孤儿temp、容量单位、关闭预算和未确认写入等确认问题的方案。两份第5节示例完成语法及本地临时数据核验：资产例只在验证中把拟改名替回现名，Runtime例使用临时SQLite持久session/transcript/result与三级缓存，关闭后可重开；这不等于拟改名已实现。

在上述声明范围、已检查代码基线和现有验证证据下，未发现尚未闭环的必须修复设计问题。已知代码缺口在第6至9节明确列入实施与回归，不将设计通过扩大为尚未实施代码、未验证生产后端或任意业务数据可丢的保证。

## 13. 缓冲写入与读取可见性

### 已确认：接纳、持久确认、读取可见性相互独立

两种路径都可以接纳尚未持久化的数据，并由同一个批写机制合并写入。区别在读视图，不在是否允许排队，也不能从读视图反推崩溃后的交付保证。

| 合同 | 可读暂存路径 | 仅已提交可读路径 |
|---|---|---|
| 写入接纳 | 先进入有界 pending，返回接纳结果 | 同样先进入有界 pending，返回接纳结果 |
| 批写 | 同一个 WriteBuffer 调度与 sink 合同 | 同一个 WriteBuffer 调度与 sink 合同 |
| 未结算读取 | 当前 owner 可读最新暂存事实；不承诺跨实例可见 | 只读权威已提交数据或其合规缓存，不返回 pending |
| sink 成功 | 确认该批次；按领域合同交接到已提交视图 | 写入方确认该批次后，才主动发布已提交缓存或推进可见 revision |
| sink 失败或结果未知 | 记录异常状态，移除已结算的暂存投影，回到权威读取；可能回退到旧值 | 不将该 pending 发布成已提交；按权威读及既有未知提交合同处理 |
| 等待 flush 超时或等待者取消 | 只结束等待，不等于 sink 已结算，不撤销仍执行的暂存项 | 同样不决定持久结果；读端仍只看已提交事实 |
| 进程崩溃 | 内存中的待写项可能丢失 | 同样可能丢失；只读已提交不等于待写可靠交付 |

第二条路径不承诺“所有写入最终成功”，也不强制同步逐条写库。如果另有“接纳后即使崩溃也必须完成”的合同，必须沿用具有 durable pending/恢复所有权的领域机制；不能仅靠 CacheBackend 或内存 WriteBuffer 宣称保证。这里不新增可靠队列或持久后台服务。

“已提交可读”不等于“调用方已收到成功响应才可读”。数据库可能已经提交而响应丢失，写调用报告结果未知时，权威读取仍可能看到新值。不能把异常当成未提交，也不能假定所有 Store 都会自动读回确认；保留各 owner 的 receipt/readback 与错误传播合同。

### 数据和缓存的发布边界

pending 与 committed 内容分别由暂存/队列和权威 reader/CacheBackend 拥有。不能把未落库值写进普通已提交缓存的同一个 key，再指望读端根据一个标志猜它是否可靠。

- 接纳只改变 pending，不改变权威 revision、已提交 metadata 或内容缓存索引
- sink 成功确认后，写入方才基于该批次主动发布已提交 revision/缓存，或使旧的可变索引失效；不可变正文仍按实际引用及摘要定位。其他权威 reader 若已读取并验证实际提交结果，即使原写调用结果未知，也可按原合同缓存该结果
- sink 失败或未知时不发布乐观副本。已存在的持久结果可能旧、可能新，读取按原权威合同处理；必要失效归原 owner，不能保留可能伪装为最新的可变副本
- 缓存发布/通知失败不回滚已经提交的数据；记录失败并使可变本地投影失效，随后按原读取策略回源
- 批次结算只移除该批次对应的 pending；同 key 后来接纳的写入仍保留。不能用 delete(key) 无条件清除新暂存值
- 已接受条目包含 queued、正在组批以及 inflight，不能只查看队列。inflight 可能已提交但未获确认，禁止直接与持久查询结果相加
- 缓存失效和更新继续遵循已有 revision/存在性/保留期合同；本方案不把既有最终一致提示暗改为全系统线性一致
- 这里的持久成功由所选权威 sink 的合同定义。把字节放入内存/可逐出文件缓存，或仅收到共享缓存写入响应，均不能自动冒充数据库已提交

WriteBuffer.confirmed 只表示回调成功返回。用于已提交读视图的 sink 必须保证成功返回已覆盖权威提交或提交结果验证；任意回调返回成功、写入另一个待写队列或可丢缓存，并不自动满足这个条件。

所有暂存读取仍需由对应领域先完成租户、身份和动态授权检查。共享一个 key-value 容器不能替代授权；缓存命中不能跳过原边界。

### 实际场景与现有 owner

| 场景 | 当前合同与读取 owner | 两条路径如何落位 |
|---|---|---|
| 自动观测指标 | 现有异步 best-effort 写入；Metrics 查询通过 MetricStore | 对应“可缓冲写、只读已提交”。首批提炼 WriteBuffer 不把 pending 自动加入聚合查询 |
| 活跃执行 transcript / interaction / payload | StagingAgentRunStore / AgentRunStore 持有同 owner 未落库事实；projection 按捕获高水位交接 | 已有领域可读暂存路径；保持其更强的恢复/终态合同，不降级成失败可丢 |
| 独立 DURABLE RuntimeHistory | archive / 权威对象 reader | 只读已提交；不保证看到另一实例尚未落库的 staging |
| revision 提示与能力元数据 | 已提交 head、RevisionSource、Capability reader | 读缓存/失效问题，不是需要批量写库的 pending 事实；继续单飞与生命周期修复 |
| 已提交正文 | ObjectStore + CacheBackend | 可丢读副本，不能混入未确认正文来提高命中率 |
| 显式允许失败的领域暂存读 | 需要由相应领域 reader 封装 pending 与 committed 的交接 | 尚需收敛通用接线；不能要求业务调用方自己拼接队列和数据库结果 |

因此，“两条路径”不需要两个 WriteBuffer 或可靠/不可靠 mode，也不能以提供 pending_items 就声称通用可读路径已完成。框架必须封装使用方所需的读取语义；原有领域 staging 可以直接复用，不能为覆盖名义新增没有实际消费者的万能 KV 存储。

### 并发、原子性和读取合并必须满足的条件

StorageWriter 的单项写与 BatchStorageWriter.apply_batch 仍保留真实提交结果及 revision 合同，不能换成队列接纳回执。原子保证仅限原 backend 支持的原子批，非原子 fallback 的部分成功结果合同仍保留。StorageChange 的 PUT、DELETE、RESET、metadata、CAS 与原子批语义保持原样：RESET 暴露下层，DELETE 是遮挡下层的 tombstone；不能把它们混成一个 None，也不能把完整业务原子批拆成多个队列条目。

简单把空 InMemoryAssetBackend 放在持久层前面不能表达通用删除暂存：删除只存在于下层的 key 可能仍让下层可读。非 primary writer 的新值也可能被更高层遮挡。因此不把任意 StorageOverlay 自动包装成乐观写缓存。

任何后续按 key 可读封装都必须处理源读跨并发变更：读取旧源开始后，同 key 可经历接纳、落库、清空 pending；源读返回时仅再看一次 pending 仍可能返回旧值。必须保留该次读取期间的变更证据，变化后重新核对或回源；证据只随未结算项/活跃读取存在，不能永久积累所有历史 key。未接纳或已结算失败项不得覆盖后来值；删除暂存必须与“没有 pending”明确区分。

append/聚合查询则不能套按 key 覆盖：inflight 与持久结果可能是同一条记录，先查询数据库再读取 pending 也可能在交接窗口漏项。稳定 ID、高水位和去重由原领域 owner 内部封装，不能丢给普通调用方，也不能靠通用 tuple 拼接解决。

### 当前公开接口结论与设计状态

保留第8节单一 WriteBuffer 的接纳、批写、flush/status/close 合同。仅已提交读视图沿现有 reader 与已提交 CacheBackend；可读暂存优先使用已有领域 owner 的读取接口。暂不新增 BufferedStorage 类、mode 参数或 pending_items 公共 API，也不将第8节纯写缓冲误称为已覆盖通用未落库读取。

本节已确认两种读取可见性及共同批写机制，修正了“第二条路径必须直接写库”的错误假定。通用可读接线还需以实际读取合同收敛并独立复审；本节不是新增 API 的实现承诺或全范围设计通过声明。实现验证须增加：pending 不污染 committed cache、提交结果未知但源可见、旧批结算遇新写、源读跨完整接纳/结算窗口、删除暂存、聚合去重/交接、不拆业务原子批与两种路径的取消/关闭行为。
