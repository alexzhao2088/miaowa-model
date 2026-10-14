# 冷启动 SOP：真实家庭数据接入标准作业流程（v2.2 成稿，ADR-0003 决策 6 衔接 PRD Phase 3）

> 定位：硬件（ESP32-S3）产出首批真实数据、真实家庭接入后，按本 SOP 执行。
> 术语见 CONTEXT.md（Domain / Weak Label / Quality Grade / Golden Dataset 等）；特征契约冻结 1.5.0。
> 前置状态：模拟器 v2 跑量与域混合加权已预埋（v2.1/v2.2），本 SOP 生效即转入真实数据节奏。

## 0. 触发条件与退出条件

- **触发**：首批真实设备开始产生 `training_features`（label_source ∈ weak_label/auto_sample，device_id 不在 `simulator_devices.txt`）。
- **退出**：按 PRD MAE 阶梯走到 W14 达标（验证集 MAE < 0.18）且灰度三步全绿，模型转常态化迭代。

## 1. 采样积累（W0 起，持续）

- 真实设备走官方数据通路落库，不做任何模拟器式干预；确认 `training_features` 按 anchor:contrast:transition ≈ 1:1.5:0.5 配比产出。
- 真实设备一律**不写入** `simulator_devices.txt`（清单外默认 real 域，零 DB 迁移）。
- 每日检查：`SELECT label_source, count(*) FROM training_features GROUP BY 1;` 确认 weak_label / auto_sample 持续增长。
- 模拟器跑量同步降速：真实数据入场后，模拟器仅作回归验证与剧本门禁数据源，不再扩量。

## 2. 标注率监控（每周）

- 口径：弱标注率 = 有弱标注（did_wake/did_cry/did_intervene 非空）的事件打点占比；打点率 = 事件场景有打点的比例。
- 阈值：打点率 < 50% 或弱标注率 < 40% 持续 3 天 → 先治标注（家长引导/打点入口优化），不硬训。
- 单设备占比 > 25% 由 `cap_single_device` 自动下采样兜底，但要在周报中点名（说明数据集中度过高）。

## 3. 训练节奏（按 PRD W8→W14 MAE 阶梯）

| 阶段 | 样本量门槛 | 达标线（验证集 20% 留出 MAE） | 动作 |
|------|-----------|------------------------------|------|
| W8 | real ≥ 1000 条 | < 0.30 | 首次真实数据训练，域混合加权开启（real ×2.0，sim = max(0.3, 1−2×real占比)） |
| W11 | real ≥ 3000 条 | < 0.24 | 复训 + 四项门禁全跑（held-out / 留一 / sq 分层 / 黄金命中率） |
| W14 | real ≥ 5000 条 | < 0.18 | 复训 + 门禁 + 灰度评审，达标进入第 4 节上线流程 |

- 每版训练：`train_xgboost.py --pg ... --domain-mix`，ONNX 交叉校验 max|Δ| < 1e-5 后才允许导出。
- 模拟器域样本随 real 占比上升自动降权；real 占比 ≥ 35% 时 sim 权重触下限 0.3，评估是否停跑模拟器。
- 任何一版四项门禁不得劣于上一版（留一 MAE 逐版对比）；劣化即回查数据质量，不带病上线。

## 4. 灰度三步上线（沿用 P5/P6 演练口径）

1. **Shadow**：`publish_model.py --mode shadow` 影子运行 ≥ 48h，`analyze_shadow.py` 一致率 ≥ 70% 方可升级。
2. **Low**：切 low 阶段，低分段不推送；误推数 = 0 且观察 ≥ 24h 方可升级。
3. **Target**：切 target 融合，融合准确率 ≥ 60%；任一指标不达标立即回滚（演练实测 3s，上限 30s）。

## 5. 异常与回退

- 真实数据断流 > 24h：排查硬件/后端管线，训练节奏冻结，不拿模拟器数据凑数冒充实绩。
- 黄金集命中率 < 60%：优先核查 `golden_events` 写入与特征窗语义，再谈调参。
- 硬件 IMU 轴定义实测后，按 ADR-0002 TODO 修正 FeatureService 姿态常量，并重训全部在架模型。
