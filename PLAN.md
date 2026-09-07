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

备注：本机宿主 Redis 抢占 6379，miaowa 指针读写一律 `docker exec miaowa-redis redis-cli`（publish_model.py 已内置）。

## 真姿态改造（2026-09-07，ADR-0002）

IMU 六轴可用 → posture 从 motion 代理升级为 accel_z 固定轴真姿态（0仰卧/1侧卧/2俯卧/3活动）；硬件固定左脚佩戴，轴映射为 3 常量（假设 z 轴仰卧 -1g），硬件到位后实测修正。契约升 1.5.0，模型 risk_model_20260907.onnx 已发布上线，e2e 51/51 全绿（新增 J1-J3）。校准制方案曾评估后放弃（硬件朝向锁死）。
TODO(硬件)：六面静置测试确认 IMU 轴定义 → 修正 FeatureService 常量。
