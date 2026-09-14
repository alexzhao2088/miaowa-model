# Miaowa Model Context

妙娃 AI 模型上下文：单 XGBoost 风险评分模型从样本构造、训练、评估、ONNX 导出到灰度上线的完整机制。代码分布：`miaowa-model/`（离线管线 + 模拟器，Python，唯一源头）+ `miaowa-backend`（在线推理与融合，Java）。

**口径权威**：《算法特征定义与数据字典 — MVP v1.4》是唯一口径来源，任何修改必须先改该文档并升 `feature_spec_version`（当前 = **1.5.0**，posture 真姿态化，见 ADR-0002）。

## Language

**Risk Score (risk_score)**:
模型输出的回归评分 ∈ [0,1]，是规则引擎的增强器，永远不单独决定用户体验。
_Avoid_: 预测值、概率、时间窗口预测

**Feature Contract (特征契约)**:
15 维特征的固定顺序与口径，绑定 `feature_spec_version`，调整必须升版本并重训。顺序：`hr_mean, hr_std, hr_trend, hrv_rmssd, motion_mean, motion_max, motion_std, posture, temp_mean, temp_trend, signal_quality_avg, time_of_day, is_night, hours_since_feeding, hours_since_sleep`。ONNX 输入名 `float_input`，shape [None, 15]。
_Avoid_: hr_max、time_since_last_event（v1 旧口径，已废弃）；day_night_flag（统一为 is_night，20:00-08:00 记 1）；time_of_day 的 sin/cos 编码（已否决，用原始小时 0-23）

**Anchor Sample (锚点样本)**:
事件前 15 分钟窗口的正样本，label=1.0。事件类型锚定风险类型：wake→wake_risk、sleep→sleep_risk、soothe/diaper→irritation_risk；**feeding 不生成锚点**，仅作时间特征。
**Contrast Sample (对照样本)**:
稳定窗口自动标签负样本，label=0.0，条件：无事件 ±30min、motion_mean < 基线×0.3、is_night=1、signal_quality ≥ 70；日上限 150 条/设备。
**Transition Sample (中间样本)**:
事件后 5-10 分钟转换区样本，label=0.3-0.5。
**Sample Ratio (样本配比)**:
anchor : contrast : transition = 1 : 1.5 : 0.5。
_Avoid_: 旧代码口径 0.9/0.1/0.5（已废弃）

**Quality Grade (质量分级)**:
A 级 = signal_quality ≥ 70 且有人工事件锚点（权重 1.0）；B 级 = sq ≥ 40 或自动标签 confidence > 0.7（权重 0.7）；C 级 = sq < 40 或 time_unreliable（不进训练）。

**Weak Label (弱标注)**:
事件打点后收集的三个布尔量 did_wake / did_cry / did_intervene，null（不确定）不参与训练。

**Golden Dataset (黄金数据集)**:
A 级样本分层抽样 10%，只用于评估、绝不参与训练。模拟期由 **Device Simulator** 的剧本真相（golden_events）替代人工复核。

**Model Shadow (模型影子)**:
灰度第一阶段：模型只算不用，`ModelShadowService` 异步落 `model_shadow_log`，离线统计与 rule_score 的方向一致率（≥70% 达标）。
_Avoid_: 与 `ShadowService`（设备影子，在线状态推导）混用——两者是不同概念

**Grayscale Phase (灰度阶段)**:
shadow → low（weight_cap 0.15）→ target（标准权重 0.3/0.15/0）三阶段，存 Redis `model:phase` + `model:weight_cap`，热切换不重启。

**Model Version Pointer (模型版本指针)**:
Redis `model:current` 指向 MinIO 中 `risk_model_YYYYMMDD.onnx`；后端启动按指针下载、校验 feature_spec 后加载；回滚 = 指针回退 + 重启（≤30s）。
_注意_：本机宿主 Redis 抢占 127.0.0.1:6379，指针读写必须走 `docker exec miaowa-redis redis-cli`（publish_model.py 已内置），直连 6379 会写到错误实例。

**Device Simulator (虚拟设备模拟器)**:
`miaowa-model/device_simulator.py`，按婴儿昼夜节律剧本经 MQTT 发 5 秒包（3-5 虚拟设备并行），剧本场景切换点即 ground truth，同时模拟家长打点（80% 打点率 / 70% 弱标注 / 0-10min 随机延迟）。

## Relationships

- 一次**事件打点**（5 类事件）触发 **Weak Label** 采集，并可生成 **Anchor Sample**（feeding 除外）；双机重复打点按 (device_id, event_type, 10min 窗口) 合并，owner 优先
- **Contrast Sample** 由自动对照采样定时产生；fallback 窗口不生成任何自动标签
- 每个样本携带 **Quality Grade**（A=1.0 / B=0.7 权重），单设备样本占比 ≤ 25%
- **Golden Dataset** 与训练集互斥；模拟期来自 **Device Simulator** 剧本真相
- **Grayscale Phase** 决定 **Model Shadow** 是否记录与 model_weight 上限
- ONNX 模型经 **Model Version Pointer** 分发，**Feature Contract** 版本不匹配拒绝加载并回退纯规则

## Example dialogue

> **Dev:** "这条锚点样本 label 打 0.9 还是 1.0？"
> **Domain expert:** "1.0——**Anchor Sample** 的口径在数据字典里写死了，0.9 是旧代码的私自 smoothing，要改掉。"

> **Dev:** "Shadow 跑了 3 天，一致率 72%，能切低权重了吗？"
> **Domain expert:** "你说的如果是 **Model Shadow**，达标了，改 Redis `model:phase=low` 就行；如果是设备影子，那跟灰度没关系。"

## Flagged ambiguities（全部已裁决）

- ~~样本 label 口径~~ → 数据字典为准：1.0 / 0.0 / 0.3-0.5，代码 0.9/0.1/0.5 作废。
- ~~"Shadow" 双重含义~~ → `ShadowService`=设备影子；`ModelShadowService`=模型影子（灰度第一阶段）。
- ~~`posture` 语义冲突~~ → ~~motion 代理~~ **二次裁决（2026-09-07）**：IMU accel 三分量 + 固定轴阈值判真姿态（0仰卧/1侧卧/2俯卧/3活动），硬件固定左脚佩戴，见 docs/adr/0002（作废 0001）。
- ~~`time_of_day` sin/cos 矛盾~~ → 保持 15 维，原始小时 0-23，字典删 sin/cos 表述。
