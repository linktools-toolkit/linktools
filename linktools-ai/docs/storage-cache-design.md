# Storage 缓存与缓冲写入统一设计

状态：已确认的框架实现与定向验收已完成；精确代码基线、测量范围和未认证环境见第9至12节。
日期：2026-10-17
范围：Issue #190 / PR #191

## 1. 目标与边界

以现有 storage 能力为基础减少重复读取、重复回源和可合并的写入。公共接口保持少量明确概念，业务调用不散落缓存 key、TTL、失效与刷新逻辑。允许对尚未发布的接口作必要破坏性调整，不保留旧别名。用户明确不要求兼容旧持久数据；不设计旧数据迁移或旧格式读取分支，也不因此授权删除实际环境数据。最新代码的功能、现有调用能力与扩展语义必须完整保留，必要接口调整同步所有调用方。

Redis 连接、配置及适配器归宿主。框架不依赖 Redis，不添加 Redis 专属参数。内存、文件、宿主共享缓存按真实复用收益组合，不强制三层。缓存不替代授权、权威存在性、保留期、事务与恢复记录。

缓冲写入与读缓存有不同职责：前者尚未获得 sink 的持久确认，不能像命中的副本一样任意逐出；后者可以丢弃并回源。本稿批写调度只用于业务明确接受延迟及异常丢失风险的数据；两种读取可见性均可使用它，不因只读已提交数据就改变交付保证。

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

### 当前收敛的公开术语

| 概念 | 统一接口 | 责任 |
|---|---|---|
| 字节缓存介质 | CacheBackend 及内存、文件、Tiered 实现 | 存取可丢 bytes，容量与介质由自身负责 |
| 领域内容缓存 | ContentCache / AssetContentCache | 领域 key 和编码，借用 CacheBackend |
| Overlay 内容注入 | content_cache | 单个领域缓存对象，不另传 adapter |
| Runtime 内容注入 | cache | 保留 Runtime 的引用、租户和来源验证 |
| 内容摘要 | digest | SHA-256 内容完整性 |
| 物理记录修订 | storage_revision / expected_storage_revision | 与业务 revision 区分 |
| 写缓存 | cache 的 append/read_batch/ack；既有 BatchWriteScope 调度 | 待写字节属于所注入实例，scope 只拥有该调用者的调度和结算，见第8节 |

其他名称只有真实职责歧义时才调整，不为了字面整齐改变公开接口。本轮不同时推行另一组未实施的路径、metadata或revision方法改名。

缓存接入使用两层职责：CacheBackend 仅保存字节，具体实现为 InMemoryCacheBackend、FilesystemCacheBackend、TieredCacheBackend；ContentCache 持有 backend 并定义 cache_key/to_cache/from_cache。AssetContentCache(backend) 直接实现该领域职责，StorageOverlay 只有 content_cache 一个缓存参数，不再另传适配器。RuntimeStorage 使用 cache，沿用自身引用、租户与源存在性校验，不新增 Runtime 适配器或 options/binding 封装。两者借用后端而不关闭它；单飞保留 value 与 encoded bytes，每个 reader 独立解码可变值。不同领域的路径与逻辑身份继续按所属合同区分，不机械给所有 DTO 添加前缀，不增加总 manager/spec/registry。

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
| 持久 history/result 大正文 | 独立预算内容缓存，复用 Runtime object router | DURABLE 才接入；字节逐次验摘要，普通观察的源成员可见性最多陈旧3秒；强结果、模型输入和展示重建仍验证当前源 |
| 短期配置/派生参数 | 复用宿主已有 TTL 与回源合并 | 失效由业务版本/刷新契约决定，不统一猜 TTL |
| 实时片段与 UI projection | 已有进程内订阅/有界状态 | 不是写库后再轮询；重连仍尊重 cursor/高水位 |
| session/graph/权限/控制 | 保留权威状态路径 | 不把租约、fence、授权、幂等当普通副本 |
| 可观测性指标/诊断 | 评估有界批写 | 先明确哪些允许延迟/丢失；记账和审计不能默认可丢 |
| 快照/恢复/冻结资产 | 各自固定读取源 | 不以最新在线缓存替代冻结内容 |

资产缓存允许同一摘要的字节跨来源复用；现有 AssetContentCache 不提供租户或物理 backend 的 key 隔离。需要独立隔离的宿主使用适配器 namespace 或独立缓存实例，不增加框架参数。Runtime 对象缓存采用另一所有者合同：key 包含 namespace、tenant、domain 与完整 ObjectRef；store_id 不是全局物理源标识；共用 cache 时同一 Runtime namespace 必须绑定同一逻辑数据源，换源须换 namespace 或 cache 实例。普通观察的正文命中无需 stat，所属读缓存 envelope 强制原始 capture 起最多3秒期限，过期回源验证存在性和完整性，L1晋升及重开 Runtime 都不续期。RECOVERY域始终保留原有当前源校验；result/wait、RuntimeHistory 结果、持久checkpoint、模型与恢复输入、展示重建（含 max_age=0）保留原有当前源 stat 验证，复用已校验字节，避免重复传输完整正文。新展示不会把旧正文期限延长；stat 能检出的变化仅遵循原 ObjectStore 合同。

## 5. 最小接入方式

已有：CacheBackend、内存/文件/Tiered实现、ObjectStore、StorageOverlay。PR191已实现并验证：RuntimeStorage.cache及内部读取路由。Storage 与 Asset 批读沿用第3节的分层合同；当前术语见第3节，已确认的写缓存公开形状在第8节。

普通调用方只注入一个可选 cache 实例，不另外构造写策略、options、总管理器或全能力包装。实例按实际需要提供完整方法组：热点 get/put/delete，批写 append/read_batch/ack，流 open_stream/read_stream。contains_many 是可选热点批探测。所属消费方仅检查自身使用的方法组；该组全部缺失就沿用该场景的直接源或已有本地流路径，半组立即 TypeError，不静默换用内部队列。宿主无需继承新 Protocol，也无需给无用能力写空方法。

SQL Runtime 在 mutation 确认提交后，可以把事务实际写入或 guard 的完整 session/execution 记录写入同一 singleton 热点 key。no-op mutation 不失效也不重发已有观察。每个 store 最多持有一个受管发布任务，忙时丢弃后续填充，不新增待写队列；缓存错误不能改变源提交结果，关闭等待该任务的取消清理。未确认或结果未知的提交不发布，整批读取仍独立捕获完整快照，不拼接 singleton。期限始终从原事务 capture 起算3秒，commit、延迟发布及跨层读取都不续期；本 owner 失效阻止旧本地填充，跨 owner 晚到的旧快照在其原期限内仍可能覆盖较新的缓存，不承诺全局单调。该优化仅覆盖这两种 SQL metadata 热点，其他 backend 保持原 read-through；节省源查询可能增加 cache put 和传输字节，须分别计量。

既有 CacheBackend 表达热点方法组，既有 StreamBackend 表达流方法组。InMemoryCacheBackend 同时提供热点和批写，共用 max_bytes/max_size；InMemoryStreamBackend 仅提供流。FilesystemCacheBackend 和 TieredCacheBackend 只提供热点。宿主已有客户端可在同一实例上同时实现任意完整组，框架不会自动向其加上另一种介质。TieredCacheBackend 可以嵌套复用热点，不引入层配置 DSL，也不承担 pending 的多层复制。

### 5.1 资产读写：内容缓存单入口

下面使用当前已落实的 content_cache、digest 与 revision source 接口。示例仅创建进程内资产，不读取、覆盖或删除实际环境数据。

    import asyncio
    import hashlib

    from linktools.ai.asset import (
        AssetContentCache, AssetKey, AssetStore, InMemoryAssetBackend,
    )
    from linktools.ai.storage import (
        StorageRevisionSource, InMemoryCacheBackend, StorageOverlay,
    )

    async def asset_example() -> None:
        backend = InMemoryAssetBackend()
        revision_source = StorageRevisionSource(backend)
        store = AssetStore(StorageOverlay(
            primary=backend,
            writer=backend,
            revision_source=revision_source,
            content_cache=AssetContentCache(InMemoryCacheBackend(
                max_size=128, max_bytes=1024 * 1024,
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

普通 store.put 已在提交后由 Overlay 调用 revision_source.revision_bumped(result.store_revision)，业务不重复通知。StorageRevisionSource 直接观察 backend.head_revision，其 revision_bumped 无需保存副本；宿主若提供共享提示适配器，仍实现 RevisionSource 的 head_revision/revision_bumped，只有适配器负责提示存储。直接绕过 Overlay 的外部写入需遵守该来源的通知/刷新合同，不能仅写提示而不提交资产。

### 5.2 已有 Runtime 历史：现有业务 API 与完整生命周期

此例使用 PR191 已有接口。runtime_root 必须是已有 RuntimeStorage.from_root 使用的同一目录，namespace、principal 及 authorization 必须对应该数据的原租户和授权策略。session_id 与 execution_id 指向可读取的既有会话和已成功完成、结果及持久 transcript 仍可读取的 Agent 执行；仅有结果而未保存 transcript 的执行不满足此例前提。本例不会创建业务记录。cache_root 是独立、专用于可丢弃副本的目录。

    from pathlib import Path

    from linktools.ai.capability import RuntimeFeatures
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
            runtime_root, cache=cache_backend,
        )
        async with RuntimeHistory.open(
            namespace,
            storage=storage,
            tenant_id=principal.tenant_id,
            features=RuntimeFeatures("history", authorization=authorization),
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

RuntimeStorage.read_object(domain, reference) 是 storage-bound SDK 读出口，供内部组装 history/transcript/result 复用；普通业务不主动调用它。它与通用 storage.read_object(store, key, ...) 的完整物化/校验工具具有不同所属对象和参数。独立冷 miss 创建新 flight，直接读取源而不先 stat；普通观察可在原 capture 的3秒内借用 flight 或热字节，晚于期限的新调用另读源，原慢调用可以正常结束。强消费者对借用 flight/热字节保留当前源 stat 验证。查询数按冷读、普通命中和强验证分别计量。禁用缓存和非 DURABLE 保留原有取消与源读取行为。

## 6. 并发、一致性与生命周期

- 同一 owning loader 内的冷 miss 共享已有 flight；共享 CacheBackend 的不同 Runtime 实例不自动获得分布式 single-flight
- completed flight 不作为新请求的缓存；权威 revision 仅允许源 I/O 开始前同批读共享，后到读必须新观察
- 已知更新的 generation/修订不能被旧刷新覆盖；opaque SDK revision 不擅自排序，宿主可按自身版本合同处理
- revision通知必须绑定实际writer的来源：现有Overlay允许非primary writer，而唯一revision_source在primary view。不能把其它层writer的head发布给primary提示源；实现按writer对应view选择source，无source则不误发，不禁止合法非primary writer，也不为此新增参数。当前实现已按此归属通知
- 取消一个共享读等待者不取消 producer；禁用缓存的原请求取消语义保持
- close 先封闭新工作，完成已接纳的必要工作，再按所有权关闭；外部 cache/连接不由 Runtime 接管
- archive/projection preflight 可以合法读取对象，必须先完成，再封闭 object router 并 drain
- 字节缓存异常/损坏可回源；RevisionSource.head_revision异常按现有契约传播，适配器只有明确返回None时才进入源回退，不新增吞异常降级。StorageOverlay.refresh 仍遵循来源策略，不是强制绕过提示的新接口；源失败、删除、授权失败不能伪装成缓存成功
- transient/volatile 不因文件或共享缓存延长生命周期；缓存逐出不是物理安全删除保证

提交结果未知与提示通知不构成同一事务：请求取消可能发生在数据库已提交而提示尚未更新的窗口。本方案保持已有请求取消/提交结果未知合同，不把关闭修复变成无条件屏蔽取消；提示仅承诺其来源定义的最终刷新行为，不承诺固定提交后新鲜度上界。需要强一致的调用继续权威读取/readback。若未来要求提示与提交原子联动，应单独设计持久通知机制，而不在缓存层偷偷加入可靠队列。

### 高频展示读取的边界

会话原始页缓存已完成原型验证，但未采用：原分页 generation 不随 execution 忙闲变化，整页复用会漏掉新的 active_execution_id；补 generation 又会让原本连续的分页新增 CURSOR_INVALID。该方案虽然热页省一次 SELECT，却还给每次实际 admission/release 增加一次批量 sequence 写入，不能当作无代价的净收益。保持原分页契约，不发布该整页缓存；已采用独立的 SessionDetail 显式 max_age 展示捕获：默认强刷新，调用方显式选择max_age=3时至多复用3秒，同时重验权限、资源存在性、active绑定与终态变化；不修改原分页generation，不用 TTL 放宽控制或授权正确性，也不另造持久影子版本。

## 7. 共享文件缓存：纯文件与共享真实用量

同一规范化 root 的所有进程通过现有 filelock 协调，对真实共享用量计量；不采用进程隔离、SQLite、新数据库依赖或进程注册表。

max_bytes 只保留在各实例内存中。每次成功写入的put持共享锁，按本实例额度对root真实用量逐出：小额度实例可压低用量，大额度实例以后仍可写高；小实例不运行时不会清理。这是用户明确接受的语义，不是永久共享最小硬上限。不持久quota，也不新增resize接口。

逐次全目录扫描原型虽通过50项定向测试，但16384项暖 put 约97.20ms，性能门未通过，不发布。目录时间戳不是可靠变更序列；锁保证互斥，却不能使旧的进程本地容量索引自动正确。

当前实现使用一个共享小计量记录，仅包含总charge和dirty状态。每次mutation持锁读取记录，先原子标dirty，再修改内容与逐出，最后原子写入clean计量。正常只stat目标及实际逐出候选，本地索引仅排热度、不信旧size；候选不足时容量压力触发扫描。每实例首次mutation无条件锁内全扫描校准，记录缺失、损坏、dirty时也扫描恢复。失败不能写假clean。没有完整共享entry索引、generation或后台服务。

取消等待mutation worker结算后才释放锁。该缓存允许丢失，不新增数据库式断电持久性保证；机器重启后首次mutation校准，不能称只读打开即完成修复。保留不受管理文件，不跟随symlink做清理，不支持旧writer绕过协议混跑。

预算继续是已提交payload逻辑计费，空内容至少1字节，并非OS实际块分配或临时文件峰值。纯文件共享计量已实现并通过定向验证，随 PR #191 发布；SQLite和持久最小quota均已否决，不是备选待确认项。

## 8. 同一 cache 实例上的缓冲批写

本节接口以本次确认的单一 cache 参数为准。普通用户只提供已有实例；不另行配置独立写缓存管理器。自动观测指标接受 best-effort 延迟和进程退出时的丢失风险；审计、恢复、控制、检查点及显式强写保持原有持久保证。数据库原生批量 API、原子事务和业务既有强 staging 不因此移除。

### 单一装配入口

    from linktools.ai.observe import Metrics
    from linktools.ai.runtime import RuntimeStorage
    from linktools.ai.storage import InMemoryCacheBackend

    cache = InMemoryCacheBackend(max_bytes=8 * 1024 * 1024, max_size=1024)
    direct = Metrics.sqlite("metrics-direct.sqlite3")
    metrics = Metrics.sqlite("metrics-cached.sqlite3", cache=cache)
    storage = RuntimeStorage.in_memory(cache=cache)

Metrics 的构造与 in_memory/sqlite/sql/from_store 工厂均接收 cache。Runtime 仍只接收 metrics，不另传指标缓存参数；同一 cache 可以供 RuntimeStorage 的热点内容与 Metrics 的自动批写按需复用。若实例仅有热点或流方法组，Metrics 直接等待源 I/O，不偷偷创建内部本地队列。

Metrics.record、record_observations 与 MetricStore.put_observations 保留确认写入合同；仅自动 recorder 走批写接纳。Metrics.query/get_observation 读取权威已提交数据，不混入 pending，不因注入缓存增加查询 TTL。

Metrics.recorder() 返回独立 MetricRecorder。每个 recorder 内部复用既有 BatchWriteScope，借用注入实例的 append/read_batch/ack 存取待写字节；领域使用现有 Observation codec 编解码及 observation_id 识别内容。不新建 DTO、公共 Policy/Protocol 或管理层。scope 的身份与接纳顺序由框架生成，普通业务不用拼 key 或 ID。

cache 的批写组接收 scope、唯一条目 ID、领域 key 与 bytes/删除值；append 异常可能已经接纳，必须暴露未知结果，不能计作确定满额拒绝或另放隐藏队列。read_batch 非破坏性返回有序未 ack 前缀，ack 只移除指定 ID。缓存不得用热点逐出语义丢弃已接纳 pending。回调成功之后 ack 失败只能重试 ack，不再次调用已成功的 sink；未能清理的 pending 必须通过 flush/close 明确暴露。

### 容量、窗口和结果

InMemoryCacheBackend 的 max_bytes 与可选 max_size 是同一实例的共同预算，包含热点项、queued、组批以及 inflight 未 ack 项。热点项可逐出，已接纳 pending 不可逐出。自动 recorder 满载拒新；StorageBuffer 的显式 enqueue 可以等待容量。单项超过后端总容量时，append 即使 wait=True 也立即返回 False；StorageBuffer.enqueue 报容量拒绝，但后续合法条目仍可接纳。仅临时满额等待，不永久挂起。append 返回 False 必须确定未接纳；接纳前取消不发布，已经接纳则先结算真实结果，不能随调用者取消遗失所有权。没有额外本地队列或“队列外再多一批”的隐藏容量。

自动 recorder 使用固定默认 128 条或首条接纳后 60 秒先到提交，后续接纳不重置首条窗口。flush 和正常 close 主动唤醒不足一批的项。60 秒是正常调度下的组批窗口，不是查询 TTL，也不包括排队和 sink 耗时。每 scope 单 writer 保序；幂等、事务与提交未知读回仍由 sink 原有领域 owner 负责。显式强写的 256 条和 16 MiB 上限保持不变。

通用状态沿用 accepting、accepted、confirmed、unconfirmed、pending、rejected、queue_size、queue_capacity、high_watermark、write_failures、last_failure_code。queue_size 包含全部未 ack 条目；容量归后端且没有统一容量查询契约，queue_capacity 为 None，不表示无界，也不能虚报独立本地额度。rejected 不计入 accepted；confirmed 只说明 sink 回调成功，不把缓存写入本身说成持久成功。指标状态 MetricRecorderStatus 将 confirmed 映射为 persisted，因为指标 sink 已确认权威提交；unconfirmed 保持未确认含义，不等于确定丢失。无 buffer 方法组的直接模式 queue_size 与 queue_capacity 为 0。pending 统计尚未结算的源结果，queue_size 统计尚未 ack 的缓存项；sink 成功后 ack 失败可以 confirmed=1、pending=0、queue_size=1，flush/close 报清理失败，不能说源未写。

### flush、关闭和取消

MetricRecorder.try_record 是异步接纳操作，bool 不能当持久提交凭据。无缓存直接模式在源调用结算后计 accepted；flush 不等待正在执行的直接调用，也不是全局写屏障。close 等待本 recorder 已进入的源调用退出，但不接管 caller 的取消。

缓冲 flush 捕获当时 accepted 前缀，主动唤醒组批并等待该前缀结算。completed 不等于所有项成功提交；状态仍保留未确认和失败。等待超时返回 completed=False；取消等待者不取消 writer，也不撤销已接纳项。

沿用 5 秒单批写预算和 10 秒关闭预算。这是停止/合作取消预算，不是任意回调的硬终止保证。callback 实际退出前仍持有工作及资源；取消请求后正常返回的真实成功仍被确认。关闭先停接纳，再排空或明确结算；不能让队尾失败掩盖前缀仍在执行。

同一 scope 的关闭有一个所有者，重复 close 共用清理结果；取消某个等待者不丢失后台工作的所有权。Runtime 等自己的 recorder 真正结束后才关闭其存储。共享 Metrics、cache、宿主连接始终是 borrowed，关闭一个 recorder 或 Runtime 不关闭它们，也不清除别的 scope 的待写项。

### 自动指标生产者的交接顺序

无缓存意味着真实新增 await。必须先完成原业务要求的交接：模型 journal/staging 与请求事件发布；工具 bridge 的成功/失败/未知结算；agent usage sink；执行终态 readback、audit、seal 和发布。随后在业务事务和锁之外等待指标。各观测在自身指标 I/O 之前冻结原业务完成时间；外层 agent/run 的 wallclock 仍可能包含内部调用的直接指标 I/O，不引入跨层扣时机制。

普通指标错误 fail-open；外部取消继续传播，但不能重写已交接的成功事实，尤其不能将成功工具误记为效果未知。Task graph/node 投影同样采用 awaited 调用，不用原后台 task 继续隐藏无缓存 I/O；事实收集与记录分别遵守自己的生命周期。

已执行的定向验证涵盖：无缓存实际等待且无隐藏队列、显式缓存接纳、128/60首条窗口、主动flush、前缀后接纳、直接并发、scope独立关闭、未知提交、组批/写入取消及成功业务交接后的指标取消。这些结果按第9节精确代码基线归档，不将设计确认替代执行证据。

## 9. 验证与完成矩阵

本次实现基线为 PR191 `69535dc05ca4b76c2e37813fabe2881cd1b7c9da`，tree 为 `c53d25217803a4981c1a2371cdbf03dde1c0ab44`。统一 cache 注入、自动 Metrics 批写和 StorageBuffer 两种读取可见性已发布，不再是本地待实现草案。该精确 head 的自然 CI run39022486641 全部9项成功，包含 Python 3.10 与 3.x 的 runtime、evaluation-recovery 和最终 gate。

2026-10-17 对相同代码 tree 补充执行348项 storage/cache/buffer/revision/当前格式与生命周期定向测试（247项9.24秒、101项5.14秒），另有126项 session/stream/backend 测试（7.67秒）及3项 Web 接线测试（1.54秒）。history result 增量的110项定向测试与上述范围有重叠，不合并为不重复的总数。定向测试不等于完整测试集；长矩阵交由上述精确 head CI。

| 已确认需求 | 实现与当前验证入口 | 结论与边界 |
|---|---|---|
| 单一 cache 接入与全局命名 | storage/asset/runtime 公开接口、`test_runtime_stream_storage.py`、架构 gate | 完整可选方法组已收敛；无新总管理器或 Redis 依赖 |
| 热点内容、miss合并、版本绑定与损坏回源 | `test_storage_cache_miss_io.py`、`test_asset_cache_integrity_and_close.py`、Runtime object/history cache 测试 | 同owner共享flight；每reader独立解码；源错误与删除不被缓存吞掉 |
| 共享纯文件真实预算 | `test_filesystem_content_cache_shared.py`、`test_storage_cache_regressions.py` | Linux本地跨进程、kill writer、dirty恢复、失败发布、空值计费和正常mutation不全扫描已验证；不是全局最小quota |
| revision新鲜度与writer归属 | `test_storage_metadata_refresh.py`、`test_storage_writer_revision_source.py` | 同批观察、后到读、旧刷新失效与非primary writer通知已验证；未引入通用TTL提示框架 |
| 当前格式与批读同步 | `test_asset_backend_conformance.py`、`test_runtime_storage_contracts.py`、`test_runtime_schema_ddl_contract.py` | digest在三backend、版本、snapshot与receipt往返；storage_revision/CAS及当前DDL验证；不增加旧格式迁移 |
| pending-visible与committed-only | `test_storage_buffer.py` | 删除、同key覆盖、旧批/新写、源读跨结算、未知/部分提交、失败回源已验证；无跨实例pending保证 |
| 容量、接纳、ack与关闭 | `test_storage_buffer.py`、`test_runtime_metric_buffer.py` | hot/pending/inflight共预算，未知接纳失败关闭，ack失败不重放成功sink，取消仍保留I/O所有权 |
| Metrics显式强写与自动批写 | `test_metrics_recording.py`、`test_runtime_metric_buffer.py`、`test_runtime_metrics.py`、`test_runtime_tool_metrics.py` | 128/60窗口、flush、close、业务交接后指标取消已验证；查询不混入pending，无聚合TTL |
| 权限、保留期与生命周期 | Runtime object/history/session、`test_storage_lifecycle_ownership.py`、recovery CI | DURABLE内容路径、权威存在性、授权、preflight/drain与借用关闭边界保留 |
| session重复刷新与stream订阅 | `test_session_display_cache.py`、`test_runtime_shared_stream.py`、`test_stream_backend.py`及Web接线测试 | 显式展示捕获、断线cursor、缺失/损坏/代际变化回退和关闭已验证；实时预览不冒充终态提交 |

### 真实读取测量

- SQLite session详情：强刷新每次6条SELECT（含正文场景8条），显式`max_age=3`冷读6/8条、热读4条。减少正文与派生读，仍保留权威校验。
- 两个真实 Runtime 使用同一 SQLite 权威源和同一个组合 cache；公开 `watch(include_content=True, include_model_interactions=True)` 在相同两秒输出窗口中分别消费1和50个实际文本delta，最终正文均为50个字符。读取方完整watch分别耗时2.403/2.394秒、均62条SELECT。测量不含最初admission/get和最终wait；不能称全栈所有SQL为62，也不承诺长流查询数不随时间增长。
- history.result 的inline读取由3降至2条SELECT，cache开启/关闭各三轮一致；精确内置repository复用授权后的fresh record，自定义与子类保留get_result。
- Runtime对象热读仍查询源成员；固定Asset版本独立调用仍验证历史源。正文传输减少与SQL减少分别计量，不以正文缓存声称消除权威SQL。

### 明确保留的边界

上游内置Web session详情仍走强history读取；这与宿主显式选择展示捕获是不同消费路径，不能据此推断所有宿主尚未接入。应用可以沿用公开refresh和watch，不接触内部cache key或队列。

`inspect_execution`固定fixture仍为10条总SELECT，其中跨owner授权不能直接删除。最后同reader重复get可进一步优化，但需保留自定义cutoff override；它不是逐文本片段查询，不构成本轮已确认契约的未修缺陷。

通用revision TTL、指标聚合TTL、长驻历史版本缓存新协议、跨进程pending视图和远端控制路由均未作为本轮新增承诺。真实Redis、Windows/网络文件系统、网络分区及生产部署不由这些本地/SQLite证据认证，不把未批准扩展列为完成阻塞项。

## 10. 实施边界及交付标准

本轮首批仅接入已有允许损失的 Runtime 自动观测契约；注入支持完整 buffer 方法组的 cache 才延迟聚合，默认直接等待源 I/O，采用明确拒新与失败不自动重试。框架能力供其他领域显式 opt-in；阶段统计等新增业务接入需各自确认，不阻塞本轮框架设计。

旧数据范围已明确：不保留旧格式兼容或迁移工具，不执行实际环境数据删除。当前代码各 backend、调用方、快照生成与读取必须同步，不能留下只改一端的状态。

其他常规命名与组装选择由设计审查负责。显式写缓存公共入口、实现及其接纳/结算/关闭回归已经独立复审并按矩阵验证；后续扩展仍需针对自己的改动重新核验。不得把静态审查或局部CI通过扩张成全部生产场景无问题保证。

## 11. 可核对来源

- 需求范围：https://github.com/linktools-toolkit/linktools/issues/190
- 已发布增量：https://github.com/linktools-toolkit/linktools/pull/191
- 已核实 CI：https://github.com/linktools-toolkit/linktools/actions/runs/39022486641
- 已发布代码基线：69535dc05ca4b76c2e37813fabe2881cd1b7c9da
- 核验文件：storage/_contracts.py、_composition.py、_cache.py、_revision.py、_lock.py；asset 各 backend；runtime/state/_root.py、_object_router.py；runtime/_metrics.py；observe/_metrics.py、_store.py、_sql.py

本稿记录已执行的精确代码基线；若后续实现改动，原结果不能替代新head验证。

## 12. 已发布范围设计复审结论

适用规则：根目录及 linktools-ai/AGENTS.md，尤其公共接口、单一所有权、revision/version 词汇、当前格式双端一致及 filelock 协调规则。审查覆盖本稿职责、名称、批读、多级缓存、修订提示、共享文件预算、写缓冲前缀与生命周期合同。

本轮已订正冷读后到借用者stat、来源提示绑定、初始化关闭、孤儿temp、容量单位、关闭预算和未确认写入等确认问题的方案。第5节原有验证使用临时资产和SQLite持久session/transcript/result与三级缓存，关闭后可重开。第8及14节实现已由第9节列出的独立定向测试补充验证，未沿用旧示例冒充新增覆盖。

在上述声明范围、已检查代码基线和现有验证证据下，未发现尚未闭环的必须修复设计问题。已确认实现范围的验证已闭环；第6至9节保留真实一致性、测量和环境边界，不将通过结论扩大为未验证生产后端或任意业务数据可丢的保证。

## 13. 缓冲写入与读取可见性

### 已确认：接纳、持久确认、读取可见性相互独立

两种路径都可以接纳尚未持久化的数据，并由同一个批写机制合并写入。区别在读视图，不在是否允许排队，也不能从读视图反推崩溃后的交付保证。

| 合同 | 可读暂存路径 | 仅已提交可读路径 |
|---|---|---|
| 写入接纳 | 先进入有界 pending，返回接纳结果 | 同样先进入有界 pending，返回接纳结果 |
| 批写 | 同一个 BatchWriteScope 调度与 sink 合同 | 同一个 BatchWriteScope 调度与 sink 合同 |
| 未结算读取 | 当前 owner 可读最新暂存事实；不承诺跨实例可见 | 只读权威已提交数据或其合规缓存，不返回 pending |
| sink 成功 | 确认该批次；按领域合同交接到已提交视图 | 写入方确认该批次后，才主动发布已提交缓存或推进可见 revision |
| sink 失败或结果未知 | 记录异常状态，移除已结算的暂存投影，回到权威读取；可能回退到旧值 | 不将该 pending 发布成已提交；按权威读及既有未知提交合同处理 |
| 等待 flush 超时或等待者取消 | 只结束等待，不等于 sink 已结算，不撤销仍执行的暂存项 | 同样不决定持久结果；读端仍只看已提交事实 |
| 进程崩溃 | 内存中的待写项可能丢失 | 同样可能丢失；只读已提交不等于待写可靠交付 |

第二条路径不承诺“所有写入最终成功”，也不强制同步逐条写库。如果另有“接纳后即使崩溃也必须完成”的合同，必须沿用具有 durable pending/恢复所有权的领域机制；不能仅靠 CacheBackend 或内存 BatchWriteScope 宣称保证。这里不新增可靠队列或持久后台服务。

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

BatchWriteScope.confirmed 只表示回调成功返回。用于已提交读视图的 sink 必须保证成功返回已覆盖权威提交或提交结果验证；任意回调返回成功、写入另一个待写队列或可丢缓存，并不自动满足这个条件。

所有暂存读取仍需由对应领域先完成租户、身份和动态授权检查。共享一个 key-value 容器不能替代授权；缓存命中不能跳过原边界。

### 实际场景与现有 owner

| 场景 | 当前合同与读取 owner | 两条路径如何落位 |
|---|---|---|
| 自动观测指标 | 现有异步 best-effort 写入；Metrics 查询通过 MetricStore | 对应“可缓冲写、只读已提交”。首批提炼 BatchWriteScope 不把 pending 自动加入聚合查询 |
| 活跃执行 transcript / interaction / payload | StagingAgentRunStore / AgentRunStore 持有同 owner 未落库事实；projection 按捕获高水位交接 | 已有领域可读暂存路径；保持其更强的恢复/终态合同，不降级成失败可丢 |
| 独立 DURABLE RuntimeHistory | archive / 权威对象 reader | 只读已提交；不保证看到另一实例尚未落库的 staging |
| revision 提示与能力元数据 | 已提交 head、RevisionSource、Capability reader | 读缓存/失效问题，不是需要批量写库的 pending 事实；继续单飞与生命周期修复 |
| 已提交正文 | ObjectStore + CacheBackend | 可丢读副本，不能混入未确认正文来提高命中率 |
| 显式允许失败的按 key 最终值读写 | StorageBuffer 复用源 get/apply_batch，内部封装 pending 与 committed 交接 | 第14节的单一视图负责接纳、覆盖、源读竞态与失败回源，不要求普通消费者拼接队列 |

因此，“两条路径”不需要两个 BatchWriteScope 或可靠/不可靠 mode，也不能以提供 pending_items 就声称通用可读路径已完成。框架必须封装使用方所需的读取语义；原有领域 staging 可以直接复用，不能为覆盖名义新增没有实际消费者的万能 KV 存储。

### 并发、原子性和读取合并必须满足的条件

StorageWriter 的单项写与 BatchStorageWriter.apply_batch 仍保留真实提交结果及 revision 合同，不能换成队列接纳回执。原子保证仅限原 backend 支持的原子批，非原子 fallback 的部分成功结果合同仍保留。StorageChange 的 PUT、DELETE、RESET、metadata、CAS 与原子批语义保持原样：RESET 暴露下层，DELETE 是遮挡下层的 tombstone；不能把它们混成一个 None，也不能把完整业务原子批拆成多个队列条目。

简单把空 InMemoryAssetBackend 放在持久层前面不能表达通用删除暂存：删除只存在于下层的 key 可能仍让下层可读。非 primary writer 的新值也可能被更高层遮挡。因此不把任意 StorageOverlay 自动包装成乐观写缓存。

pending-visible 按 key 可读封装必须处理源读跨并发变更：读取旧源开始后，同 key 可经历接纳、落库、清空 pending；源读返回时仅再看一次 pending 仍可能返回旧值。必须保留该次读取期间的变更证据，变化后重新核对或回源；证据只随未结算项/活跃读取存在，不能永久积累所有历史 key。未接纳或已结算失败项不得覆盖后来值；删除暂存必须与“没有 pending”明确区分。

append/聚合查询则不能套按 key 覆盖：inflight 与持久结果可能是同一条记录，先查询数据库再读取 pending 也可能在交接窗口漏项。稳定 ID、高水位和去重由原领域 owner 内部封装，不能丢给普通调用方，也不能靠通用 tuple 拼接解决。

### 当前公开接口结论与设计状态

第8节 BatchWriteScope 保留 append 接纳、批写和结算合同。通用按 key 最终值视图由第14节 StorageBuffer 提供，复用同一调度机制；两种读取模式在构造时选择，不暴露 pending_items 让消费者自行合并。既有强 staging 继续由领域 owner 持有，不能以通用可丢缓冲替代。

两种读取可见性都支持共同批写机制，仅已提交可读不要求同步逐项写入。第14节的最终值缓存和第8节的 append 缓冲保持不同的数据语义。验收必须覆盖：pending 不污染 committed cache、提交结果未知但源可见、旧批结算遇新写、源读跨完整接纳/结算窗口、删除暂存、聚合去重/交接、不拆业务原子批与两种路径的取消/关闭行为。

## 14. 通用按 key 可读缓冲

`StorageBuffer(store, read_pending=False, cache=cache)` 为已存在的
字节值 store 增加独立的跨调用缓冲视图。store 继续通过公开 `get` / `apply_batch`
提供权威读取和批写；不要求另一套持久适配器，不替代 `StorageWriter` 的持久回执、
CAS、RESET 或领域原子事务。普通消费者只操作一个 buffer：`enqueue` /
`enqueue_delete` 接纳、`get` 读取、`flush` 确认及 `close` 排空。

接纳容量不足时等待，关闭或失败明确报错；返回不代表已持久化。读取模式在装配时
选择一次：默认仅返回源已确认持久化的数据，不表示每次都读取源的最新值。存在完整
热点能力时，committed-only 也可命中原 capture 起最多3秒的热快照；命中、层间提升
均不续期，其他实例或外部写入的变化可在该窗口内延迟可见。启用 `read_pending` 时，
最新未结算值或删除优先覆盖该已确认视图，不能把 pending 当作 committed。
同 key 同批可只写最后意图，不能把中间版本未落库称为数据丢失。Metrics 的
append 记录不采用这种覆盖合并。

`flush` 成功表示调用前意图已落库，或已被之后成功落库的同 key 值覆盖；不是精确
前缀快照或每个中间版本的持久承诺。首个写失败或结果未知使 buffer 停止接纳新写，
读取回源，`flush` / `close` 明确失败；不自动重放 sink，不增加业务 receipt/reset 系统。
已经部分提交的事实仍由源决定，不把异常当作回滚。

并发正确性归 buffer：pending-visible 源读取跨完整接纳、提交和移除 pending 的窗口必须重验；
committed-only 的冷读保留该次源读取的快照合同，不追踪等待期间的后续提交；有效
hot 命中按原3秒期限返回已确认快照。成功结算仅在本 owner 失效旧hot，下次读取一次
源再填充，回填不能改写持久回执或阻塞ack。旧批结算不能删新值；未 dispatch 队列在关闭时也必须结算清理。pending 与活跃读取的
代次记录仅保留必要生命周期，不永久积累历史 key。使用同一不可变 bytes 引用，待写数据保留在注入 cache 中；其预算包含 inflight。逻辑 payload 计费不是进程 RSS 或物理磁盘硬上限。

源为借用资源。关闭先停止新接纳，等待实际写入和已接纳读取退出后才能允许源关闭；
取消等待者不能丢失内部任务所有权。同 key 写入不得绕过 buffer 与旧队列竞争。
读取仍须先由领域执行授权。已确认hot命中不再回源证明；miss 才调用源并遵循源的
ContentCache、metadata及校验合同。显式强结果/控制读取不通过这个普通hot视图。

这项通用能力不要求 ExecutionStatus 改为可丢状态，不接管 Runtime 事务内 identity
map、原生批 SQL、checkpoint、取消、恢复和终态提交。它补充的是两种读取可见性与
最终值的跨调用聚合；源的强一致责任保持在原 owner。
