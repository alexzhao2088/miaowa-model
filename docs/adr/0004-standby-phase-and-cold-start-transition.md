# ADR-0004: 冷启动衔接期训练策略——待命期五项工作 + 真实数据期门禁口径

## 状态

已接受（2026-10-15）。前序：ADR-0003（模拟器 v2 + 域混合加权预埋，已交付 v2.1/v2.2）。

## 背景

ADR-0003 的 4 周窗口（2026-09-17 起）于 2026-10-15 期满，v2.1/v2.2 均提前于 2026-10-11 完成，五项门禁全过（v2.2：验证集 MAE 0.1016、held-out 差值 0.0516、LODO 0.2124、sq 差值 0.0273、黄金 98.02%，split contract v2）。按 ADR-0003 决策 6 的出口约定：硬件未产出首批真实数据 → 转冷启动待命，不空转训练。

当前事实基线（2026-10-15 核查）：

- 本地已发布：`model:current = risk_model_20261011_dm.onnx`，`model:phase = target`（P6 演练口径延续）。
- `training_features` 共 786 条，全部来自模拟器/e2e/测试设备，**无真实婴儿数据**；真实形态设备（MW2026080001-10 等，2026-09-24 注册）仅有台架测试样本（9 月 cry_detect/alert_feedback 零星条数）。
- 生产云部署进行中（见 miaowa-deploy-handoff.md）：MinIO 容器未修复，生产在架模型仍是旧版 `miaowa_xgb.onnx`，模型分发链路未通。
- 模拟器自 2026-10-11 后停跑，管线进入静置。
- 遗留缺口：`assign_weights` 只认 `weak_label` 为 A 级人工锚点，`cry_detect`（后端注释：置信度高于 weak_label 人工打点）与 `alert_feedback`（家长回访确认/误报）被降为 B 级 0.7，与后端语义不一致。

## 决策

1. **衔接期两段式**：Phase S（待命期，现在 → 真实数据入场）只做五件事——S1 发布与生产分发收口、S2 训练语义补齐、S3 公开数据集特征校准、S4 模拟器降频回归、S5 真实数据期待办固化；Phase C（冷启动执行期）按 docs/cold-start-sop.md 执行，本计划仅补充微调策略层细节。**S 期不出新上线模型**（回归模型版本记号 v2.2-rN，仅留档），真实数据首版模型起 **v3.0**（W8）→ v3.1（W11）→ v3.2（W14）。
2. **cry_detect / alert_feedback 升 A 级**：`assign_weights` 的 A 级（sq≥70 → 权重 1.0）适用范围从 weak_label 扩至三类确认型标注——weak_label（人工锚点）、cry_detect（哭声强标注，哭声接入方案决策 9 定为 label=1.0 强标注，后端注释置信度高于人工打点）、alert_feedback（家长回访 confirmed/false_positive 直接写 label，sample_type 随反馈分 anchor/contrast）。
3. **公开数据仅校准、不进训练**（沿用 ADR-0003 决策 7 边界，落地调研 P0 两项）：picsdb（ODbL 1.0，仅内部使用不触发 ShareAlike）校准模拟器 hr/hrv 生成器参数；wearable-sensors-infant（CC BY 4.0，踝部子集优先）校准 sq 分布先验与 SQI→signal_quality_avg 映射。产出为参数表与校准报告，**不产出训练样本、不改特征契约**。
4. **真实数据期门禁口径**：恢复 PRD 绝对 MAE 阶梯为主门禁（W8 <0.30 / W11 <0.24 / W14 <0.18，验证集 20% 留出）；v2.x 四项相对门禁（held-out 剧本差值 / LODO / sq 分层 / 黄金命中率）保留全跑但转辅助监控——其中 **held-out 剧本门禁在 real 样本占比 ≥50% 后降为报告项**（剧本是模拟器产物，真实数据期其一票否决意义消失）。任何一版不得显著劣于上版（回归 margin 0.05）。
5. **契约冻结与重训触发器**：特征契约继续冻结 1.5.0；硬件到位后两项后端修复触发契约 1.5.1 与全量重训——IMU 六面静置测试修正 FeatureService 姿态常量（ADR-0002 TODO）、AutoLabelService 对照条件修复（motion<基线×0.3 不可达 → 与 harvest_contrast.py 绝对静止阈值口径对齐）。

## 后果

- 交付物：PLAN.md 衔接期章节、本 ADR、CONTEXT.md 质量分级口径更新 + 2 新术语（Standby Phase / Feature Calibration Domain）、train_xgboost.py assign_weights 修复（S2）。
- 飞书《AI模型落地类文档》本阶段不同步（同 ADR-0003 口径）；Phase C 门禁口径实际切换、契约升 1.5.1 时同步修订飞书侧数据字典与落地文档。
- 生产环境模型分发（S1）依赖部署线 MinIO 修复，操作序列以 miaowa-deploy-handoff.md 为准，本计划不越权执行。
- 备选方案及否决理由：S 期继续扩量模拟器跑版本（ADR-0003 已封顶"不空转训练"，且剧本池边际收益已尽——黄金命中率 98% 接近饱和，否决）；S 期接入公开数据直接混合训练（域差异大、样本量不足、许可约束，调研结论已否决，维持校准定位）；立即修复后端两项 Java 缺陷（真实数据入场前无验证载体，且契约冻结期不应动后端，推迟到 Phase C 第一周）。
- 风险残留：picsdb 为早产儿 ICU 数据，与足月家庭婴儿分布有差，校准时分层使用并在报告中注明；S3 校准改变模拟器分布后，门禁 1（MAE 回归监控）基线需在 eval_history 注明重置。
