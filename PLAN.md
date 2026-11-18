# 妙娃 AI 模型独立开发调试计划 — 灰度就绪版

> 依据文档：《PRD — MVP v1.4》《AI模型落地文档 v1.4》《算法特征定义与数据字典 v1.4》（口径权威）《数据标注规范 v1.4》
> 目标：硬件与真实用户到位前，独立完成模型全链路开发与调试，达到「灰度三步」可随时对真实数据上线的状态
> 术语口径见 [CONTEXT.md](./CONTEXT.md)；关键决策见 [docs/adr/](./docs/adr/)

## 已裁决决策（拷问结论）

| # | 决策点 | 结论 |
|---|--------|------|
| 1 | 计划边界 | 灰度就绪版：训练管线 + 模型 Shadow + 灰度切换 + 版本管理，不含真实用户线上评估 |
| 2 | 代码组织 | 新建 `miaowa-model/` 平级文件夹，`miaowa-repo/model/` 全量迁入为唯一源头 |
| 3 | 特征契约 | 以数据字典 15 维为准，`feature_spec_version=1.4.0`，ONNX 输入 `float_input` [None,15] |
| 4 | posture | motion_mean 代理（still/low/high→0/1/2），见 ADR-0001 |
| 5 | time_of_day | 原始小时 0-23 占 1 维，字典删 sin/cos 表述 |
| 6 | 标签口径 | 正 1.0 / 负 0.0 / 中间 0.3-0.5（代码 0.9/0.1/0.5 作废） |
| 7 | 模型 Shadow | 新建 `ModelShadowService` + `model_shadow_log` 表 |
| 8 | 灰度切换 | Redis `model:phase` + `model:weight_cap`，热切换不重启 |
| 9 | 版本管理 | MinIO `risk_model_YYYYMMDD.onnx` + Redis `model:current` 指针，回滚=指针回退+重启 ≤30s |
| 10 | 模拟数据源 | 虚拟设备模拟器（昼夜剧本，3-5 设备）+ 剧本真相黄金集 + 模拟家长打点 |

## P0 地基对齐（约 0.5 天）

- 迁移 `miaowa-repo/model/` 4 文件 → `miaowa-model/`，同步修改后端容器 `-v` 挂载路径与 `MIAOWA_MODEL_PATH`
- 数据库迁移脚本：`training_features` 增 `hours_since_feeding`/`hours_since_sleep`/`is_night` 列、废弃 `hr_max`/`time_since_last_event`；新建 `model_shadow_log` 表；补录 schema_version
- ~~人工项：修订飞书数据字典~~ ✅ 已通过飞书 API 完成（posture 改 motion_mean 代理注明、time_of_day 删 sin/cos 表述）

## P1 特征层口径改造（后端 Java，约 1.5 天）

- `FeatureService` 按字典重写：`hr_trend`=近15min均值−前30min均值；`temp_mean` 15min 窗；`temp_trend` 30min 斜率；`is_night` 20:00-08:00；`hours_since_feeding/sleep`（封顶 12，缺省 12，源自 event_record）；posture 代理 0/1/2；缺失值按字典策略填充（基线/0/NaN）
- `ModelInferenceService.FEATURE_ORDER` 同步为新 15 维顺序
- 验收：e2e 特征断言更新后全绿；Redis `features:*` hash 与字典逐字段对得上

## P2 标签与样本构造口径（后端 Java，约 2 天）

- `AnchorLabelService`：label 改 1.0；feeding 不生成锚点；事件类型→风险类型映射（wake/sleep/soothe/diaper→irritation）；中间样本生成（事件后 5-10min，label 0.3-0.5）
- `AutoLabelService`：稳定窗口四条件（无事件±30min、motion<基线×0.3、is_night=1、sq≥70）；日上限 150/设备；补传样本置信度上限 0.8 且标记 backfilled；fallback 窗口不生成
- `EventController`：双机打点 (device_id, event_type, 10min 窗口) 合并，弱标注冲突 owner 优先、无 owner 取「是」
- 验收：e2e E9 断言改 label=1.0 并通过；新增双机合并测试

## P3 训练管线升级（miaowa-model/ Python，约 2 天，可与 P1/P2 并行）

- `train_xgboost.py`：15 维新契约；ONNX 输入名 `float_input`；固定超参 100/4/0.1/0.8；改 skl2onnx；A/B 级加权 1.0/0.7；单设备占比 ≤25%；stratify 分层；MAE 不达标禁止导出
- `evaluate.py`：验证集 MAE + 黄金集高分段(>0.7)命中率（≥60%）+ 补传 vs 实时分层 MAE 差值（<0.05）
- `publish_model.py`：ONNX 按日期命名上传 MinIO、记录 feature_spec_version、更新 Redis `model:current`
- 验收：合成数据全流程跑通，产出评估报告；MinIO 可见版本、指针正确

## P4 虚拟设备模拟器（miaowa-model/ Python，约 2 天）

- `device_simulator.py`：婴儿昼夜节律剧本（深睡/浅睡/清醒/哭闹场景切换 + 信号质量波动 + 偶发佩戴松动/脱落），3-5 虚拟设备并行，5s 一包，7×24 可持续
- 剧本真相导出 `golden_events`（场景切换时间戳 = ground truth）
- 模拟家长：80% 概率打点、70% 概率答弱标注、0-10min 随机延迟、双机同时打点
- 验收：跑 24h 后 `training_features` 三类样本配比 ≈1:1.5:0.5、负样本日产 >100/设备、自动标签一致率可统计

## P5 Shadow 与灰度机制（后端 Java + 分析脚本，约 2 天）

- `ModelShadowService`：融合层每包异步落 `model_shadow_log`（model_score/rule_score/final_score/sq/model_version/agree_flag）
- 融合层读取 Redis `model:phase`/`model:weight_cap`（短 TTL 缓存），phase=shadow 时模型只算不用
- `ModelInferenceService`：启动按 `model:current` 从 MinIO 下载模型、校验 feature_spec_version（不匹配拒载回退纯规则并告警）、连续 10 包推理异常自动降级纯规则
- `analyze_shadow.py`：方向一致率统计（达标线 ≥70%）
- 验收：phase 切换秒级生效；错误 feature_spec 模型被拒载；MinIO 断连时回退纯规则

## P6 灰度三步演练（约 4 天，含观察）

1. **Shadow**：模拟器全开 3 天（先 6h 压缩冒烟）→ 方向一致率 ≥70%
2. **低权重**：`model:phase=low`（cap 0.15）4 天（可 1 天冒烟）→ 误推送率不升、模拟家长零投诉
3. **目标权重**：`model:phase=target`（0.3/0.15/0）→ 融合准确率 ≥60%（对照 golden_events）
4. **回滚演练**：`model:current` 回退上一版 + 重启 ≤30s，验证旧版正常加载

## 依赖与执行顺序

```
P0 → P1 → P2 ─┐
P0 → P3 ──────┤（P1/P2 与 P3 可并行：Java vs Python）
              ↓
             P4 → P5 → P6
```

## 风险与注意

- **合成数据 MAE 虚低**：seed_synthetic 特征与 label 强相关，MAE≈0 不代表真实水平，评估报告必须注明，真实数据到位后重估
- **模拟器剧本过拟合**：剧本需加噪声与随机事件，防模型学成"剧本识别器"
- **e2e_test.py 全程同步**：特征名、label 断言（E9）、Redis key 随 P1/P2/P5 同步更新
- **飞书文档修订为人工项**：数据字典 posture/sin-cos 两处修订需在飞书侧完成，保持口径权威一致

## 执行进度（2026-09-06）

| 阶段 | 状态 | 关键结果 |
|------|------|----------|
| P0 地基 | ✅ | miaowa-model/ 迁移、迁移脚本 20260905_001（新列+model_shadow_log+golden_events）已执行 |
| P1 特征层 | ✅ | FeatureService 15 维新口径上线，e2e 全绿 |
| P2 标签口径 | ✅ | 锚点 1.0/对照 0.0/中间 0.3-0.5/双机合并，e2e E9-E11 全绿 |
| P3 训练管线 | ✅ | 自研 onnx_export.py（弃 onnxmltools，IR8，交叉校验 max\|Δ\|≈6e-7）；合成数据 MAE 0.0177、黄金命中率 100%；publish/rollback 实测通过 |
| P4 模拟器 | ✅ | 修复 device 表外键丢弃；压缩跑量：45 真相/34 打点/样本正常产出 |
| P5 Shadow/灰度 | ✅ | e2e 48/48（含 I1-I3 灰度用例）；spec=9.9.9 模型拒载回退纯规则验证通过；analyze_shadow.py 达标判定 PASS |
| P6 灰度演练 | ✅ | Shadow 一致率 80.7%≥70%；low 阶段误推 0 不升；target 融合准确率 83.3%≥60%；回滚 3s≤30s；e2e 48/48 全绿 |

| v2.1 模拟器 v2 + 门禁（2026-10-11 完成） | ✅ | 四波跑量（含夜间 22:00-06:00 对照窗 + 白天补多样性波），样本 766 条（锚点 297/过渡 286/对照 183，held-out 设备 184 条）；出模型 risk_model_20261011.onnx，五项门禁全过：held-out 差值 0.0479<0.08、LODO 0.2124 不劣化、sq 分层差值 0.0243<0.05、黄金命中率 98.02%≥60%、验证集 MAE 0.1072（契约 v2 基线）。关键修复：发现并修复验证集划分契约不一致缺陷（cap_single_device 重排行序导致 evaluate.py 验证集落在训练集内、口径虚低，0.0492 vs 诚实口径 0.1072），make_val_split 升为两侧共用（split contract v2），load_data 增加 ORDER BY 定序；旧评估记录口径不可比，门禁 1 基线重置。已知缺陷遗留：后端 AutoLabelService 对照条件 motion<基线×0.3 不可达（基线 25s 跟踪睡眠体动，MW-TEST 三周 0 条佐证），离线采集器 harvest_contrast.py 以绝对静止阈值 0.014 替代，后端修复列入真实数据阶段待办 |
| v2.2 域混合加权（2026-10-11 完成） | ✅ | --domain-mix 开关上线（real ×2.0，sim=0.85）+ --pseudo-real-device 自测开关 + 域分层报告；出模型 risk_model_20261011_dm.onnx，五项门禁全过且全面不劣于 v2.1（验证集 MAE 0.1016、held-out 差值 0.0516、LODO 0.2124、sq 差值 0.0273、黄金 98.02%，real 域 MAE 0.0221）；冷启动 SOP 成稿（docs/cold-start-sop.md）；公开数据集调研完成（结论：无集可直接预训练，picsdb + wearable-sensors-infant 可作特征校准域）。发布上线（publish_model.py 指针切换）留待人工拍板 |
| 衔接期启动（2026-10-15，ADR-0004） | 🔄 | 4 周窗口期满、真实数据未入场 → 按出口约定转待命。ADR-0004 与 PLAN 衔接期章节成稿（Phase S 五项 + Phase C 微调策略层，真实数据模型起 v3.0）；S2 训练语义补齐落地：assign_weights A 级扩至 weak_label/cry_detect/alert_feedback 三类确认型标注，test_pipeline 8/8 全绿；本地 model:current=risk_model_20261011_dm.onnx、phase=target 已确认。进展（2026-11-18）：S1 已收口（dm 模型生产上线，按指针加载确认）；S3 已完成 picsdb 心率分布校准（9317 个 5min 窗，docs/calibration/；结论：心动过缓富集库绝对中位数不可用，模拟器区间落在真实包络内，不调整 BASE_RANGES；wearable-sensors-infant sq 先验待数据集渠道）；S4 每周回归机制落地（weekly_regression.sh + launchd 周一 03:07，栈未就绪自动 SKIP）；Phase C 触发值守落地（phase_c_watch.py + launchd 每日 08:17，排除模拟器/测试设备且仅统计近 14 天样本）；S5 两项后端修复列入 Phase C 第一周 |

备注：本机宿主 Redis 抢占 6379，miaowa 指针读写一律 `docker exec miaowa-redis redis-cli`（publish_model.py 已内置）。

## 真姿态改造（2026-09-07，ADR-0002）

IMU 六轴可用 → posture 从 motion 代理升级为 accel_z 固定轴真姿态（0仰卧/1侧卧/2俯卧/3活动）；硬件固定左脚佩戴，轴映射为 3 常量（假设 z 轴仰卧 -1g），硬件到位后实测修正。契约升 1.5.0，模型 risk_model_20260907.onnx 已发布上线，e2e 51/51 全绿（新增 J1-J3）。校准制方案曾评估后放弃（硬件朝向锁死）。
TODO(硬件)：六面静置测试确认 IMU 轴定义 → 修正 FeatureService 常量。

## 下一阶段：模型微调训练计划（2026-09-17 起，4 周窗口，ADR-0003）

> 定位：硬件/真实婴儿到位前，以模拟器 v2 + 域混合加权预埋保持管线热度，交付真实数据冷启动 SOP 衔接 PRD Phase 3。特征契约冻结 1.5.0。
> 术语：Domain / Held-out Script Pool / Pseudo-real Domain / Cold-start SOP 见 [CONTEXT.md](./CONTEXT.md)。

### 验收口径（替代 PRD 绝对 MAE 阶梯，真实数据到位后恢复 PRD 口径）

| 门禁 | 口径 | 达标线 |
|------|------|--------|
| Held-out 剧本测试（核心新增） | 模型在 20% 未参与训练的剧本上的 MAE | 与训练池剧本 MAE 差值 < 0.08 |
| 跨设备留一验证 | leave-one-device-out MAE | 报告并逐版不劣化 |
| 信号分层鲁棒性 | sq 40-70 区间 vs sq≥70 区间 MAE 差值 | < 0.05（沿用补传分层口径） |
| 黄金集高分段命中率 | >0.7 分段命中 golden_events 比例 | ≥ 60% |
| 绝对 MAE | 验证集 20% 留出 | 仅回归监控，不许显著劣于上版 |

### 里程碑

| 版本 | 时间 | 内容 | 出口 |
|------|------|------|------|
| v2.1 | W2 末（~10-01） | ① 模拟器 v2：剧本池参数化（规律型/夜醒频繁型/小睡碎片化型，随机种子）+ 设备个性化（基线漂移/松动/脱落）+ 异常场景（仅现有 5 事件 + sq 波动）② 剧本池 8:2 切训练/Held-out ③ `simulator_devices.txt` 域标记清单 ④ 跑量 5 设备×7 天 ⑤ evaluate.py 接入四项门禁 | 出模型 v2.1，四项门禁全过 |
| v2.2 | W4 末（~10-15） | ① train_xgboost.py 混合加权开关（real ×2.0，sim = max(0.3, 1−2×real占比)）② Pseudo-real Domain 自测加权与分层报告 ③ 冷启动 SOP 成稿 ④ 公开数据集调研结论（3 天时间盒，本阶段不接入） | 出模型 v2.2 + SOP + 调研结论 |

阶段出口（取先到者）：硬件产出首批真实数据 → 立即转冷启动 SOP 执行；或 4 周期满 → 转冷启动待命，不空转训练。

### 边界与注意

- 后端与 e2e（58 用例）不动；改动集中在 miaowa-model/ Python 侧，自测覆盖。
- 飞书《AI模型落地类文档》本阶段不同步；真实数据阶段口径实际修订时再同步。
- 风险残留：模拟器 v2 剧本池仍是人工分布，与真实婴儿的 gap 只能延迟到冷启动阶段用真实数据重估（ADR-0003）。
- 公开数据集调研结论追加到本节附录（W1 内完成）。

## 下一阶段：冷启动衔接期微调训练计划（2026-10-15 起，ADR-0004）

> 定位：v2.2 交付、ADR-0003 四周窗口期满，硬件/真实婴儿仍未到位 → 按出口约定转**冷启动待命**。本阶段不虚训、不空转：待命期（Phase S）收口遗留缺口并保持管线热度，真实数据入场即切冷启动执行期（Phase C，SOP 主体已就位，本节补微调策略层）。
> 决策记录见 [docs/adr/0004](./docs/adr/0004-standby-phase-and-cold-start-transition.md)；术语见 [CONTEXT.md](./CONTEXT.md)（Standby Phase / Feature Calibration Domain）。

### 事实基线（2026-10-15 核查）

- 在架模型 v2.2 `risk_model_20261011_dm.onnx`：五项门禁全过（MAE 0.1016 / held-out 差值 0.0516 / LODO 0.2124 / sq 差值 0.0273 / 黄金 98.02%），本地 `model:current` 已指向、`model:phase=target`。
- 样本库 786 条（weak_label 402 / auto_sample 338 / cry_detect 19 / alert_feedback 7），全部 sim/e2e/测试域，**零真实婴儿样本**。
- 生产云部署进行中：MinIO 未修复，生产在架仍为旧版 `miaowa_xgb.onnx`，模型分发链路未通（见 miaowa-deploy-handoff.md）。
- 模拟器 2026-10-11 后停跑。

### Phase S：待命期（现在 → 真实数据入场）

| # | 工作项 | 内容 | 时间盒 | 出口 |
|---|--------|------|--------|------|
| S1 | 发布与生产分发收口 | 部署线 MinIO 修复后：`risk_model_20261011_dm.onnx` 上传生产 MinIO models bucket + 生产 Redis `model:current` 切换；生产后端校验 feature_spec 1.5.0 匹配可加载 | 随部署序列 | 生产 `model:current=dm` 且后端加载成功 |
| S2 | 训练语义补齐 | `assign_weights` A 级扩至三类确认型标注（weak_label / cry_detect / alert_feedback，ADR-0004 决策 2）；test_pipeline.py 增补分级用例 | 2026-10-15 | ✅ 已完成（代码+自测全绿） |
| S3 | 公开数据集特征校准 | picsdb（ODbL，内部使用）→ RR 间期提取 → 0-12 月龄 hr_mean/hr_std/hr_trend/hrv_rmssd 分布表 → 校准模拟器心率生成器；wearable-sensors-infant（CC BY 4.0，踝部子集优先）→ sq 分布先验 + SQI→signal_quality_avg 映射。**仅校准不进训练**（ADR-0004 决策 3） | 2 周（~10-29） | 校准参数表 JSON + 模拟器参数更新 + 校准报告（追加调研文档） |
| S4 | 模拟器降频回归 | 每周 1 次 24h 压缩跑量（7 设备）→ harvest_contrast → train → evaluate 五项门禁留档（版本记号 v2.2-rN，**不上线**）；S3 校准落地当周出对比模型留档；门禁回归失败即排查 | 持续每周 | eval_history 连续留档、无未解释 FAIL |
| S5 | 真实数据期待办固化 | ① IMU 六面静置测试 → FeatureService 姿态常量修正 → 契约 1.5.1 → 全量重训（ADR-0002 TODO）② AutoLabelService 对照条件修复（motion<基线×0.3 不可达 → 对齐 harvest_contrast 绝对静止阈值口径）。两项均为后端 Java 改动，列入 Phase C 第一周 | 硬件到位即做 | 列入 Phase C 启动清单 |

### Phase C：冷启动执行期（真实数据入场触发，主体 = docs/cold-start-sop.md）

触发判定（SOP 第 0 节）：非 `simulator_devices.txt` 设备产生 label_source ∈ {weak_label, auto_sample, cry_detect, alert_feedback} 的 `training_features`。微调策略层补充：

| # | 策略 | 口径 |
|---|------|------|
| C1 | 版本线与主门禁 | 真实数据模型起 **v3.0**（W8，real ≥1000，MAE <0.30）→ v3.1（W11，real ≥3000，<0.24）→ v3.2（W14，real ≥5000，<0.18）；PRD 绝对 MAE 阶梯恢复主门禁（ADR-0004 决策 4） |
| C2 | 辅助门禁 | v2.x 四项相对门禁每版全跑转辅助监控；**held-out 剧本门禁在 real 占比 ≥50% 后降报告项**；任何一版不得显著劣于上版（margin 0.05） |
| C3 | 域权重衰减 | `--domain-mix` 公式自动衰减（real ×2.0，sim = max(0.3, 1−2×real占比)）；real ≥35% sim 触下限 0.3 时评估模拟器退出；real ≥1000 且连续两版门禁全过 → 模拟器转纯回归/门禁数据源 |
| C4 | 黄金集双轨 | 真实黄金集（A 级确认型标注分层抽 10%，只评估不训练）与剧本真相 golden_events 并存报告；真实黄金集 n≥100 后以真实口径为准 |
| C5 | 契约 1.5.1 重训 | S5 两项后端修复落地即升 feature_spec_version=1.5.1（维度不变），全部在架模型重训重评估，eval_history 注明口径切换 |
| C6 | 标注率监控 | SOP 第 2 节：打点率 <50% 或弱标注率 <40% 持续 3 天 → 先治标注不硬训；单设备占比 >25% 周报点名 |
| C7 | 灰度三步 | SOP 第 4 节：Shadow ≥48h（一致率 ≥70%）→ low ≥24h（误推 0）→ target（融合准确率 ≥60%）；PRD Phase 3 验收：MAE <0.2、自动标签 >100 条/设备/日 |

### 边界与风险

- 特征契约冻结 1.5.0（Phase S）；1.5.1 仅为硬件常量补丁（维度不变，ADR-0004 决策 5）。
- Phase S 不出新上线模型；后端 Java / e2e（58 用例）不动；生产环境操作走部署交接序列，本计划不越权执行。
- 飞书《AI模型落地类文档》本阶段不同步；Phase C 门禁口径切换与契约 1.5.1 时同步修订飞书数据字典/落地文档。
- 风险：picsdb 早产儿 ICU 分布与足月家庭婴儿有差（分层校准、报告注明）；ODbL ShareAlike（仅内部使用不触发，处理产物不对外分发）；S3 校准改变模拟器分布后门禁 1 基线需在 eval_history 注明重置；待命期过长导致管线锈蚀由 S4 每周回归兜底。
