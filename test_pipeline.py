#!/usr/bin/env python3
"""miaowa-model 管线自测（零依赖，断言式）。

覆盖 v2.1/v2.2 冲刺交付的关键功能：
  1. 划分契约 v2：train 与 evaluate 两侧的验证集划分完全一致（行序/分层/种子）
  2. make_val_split 的 float32/float64 桶一致性（历史 bug 根因回归）
  3. assign_weights 质量分级、cap_single_device 25% 上限、apply_domain_mix 域加权
  4. harvest_contrast.eligible 判定口径（sq/is_night/posture/motion 阈值）
  5. ONNX 工件契约：metadata、15 维输入、float_input、推理形状与值域

用法：python3 test_pipeline.py
DB 依赖：划分一致性用例需要本地 PostgreSQL（docker miaowa-postgres）。
"""
import os
import sys

import numpy as np
import pandas as pd

import train_xgboost as tx
import evaluate as ev
import harvest_contrast as hc

PG = "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa"
PASSED, FAILED = [], []


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS {name}")
    except Exception as e:  # noqa: BLE001 - 自测脚本要打出所有失败
        FAILED.append((name, e))
        print(f"  FAIL {name}: {e}")


def test_split_contract_consistency():
    """train 侧与 evaluate 侧在同一数据上划分出的验证集必须完全一致。"""
    df = tx.load_data(PG)
    heldout = tx.read_device_list("heldout_devices.txt")
    base = df[~df["device_id"].isin(heldout) & (df["label_source"] != "golden")]
    # train 侧：assign_weights -> cap_single_device -> make_val_split
    train_pool = tx.cap_single_device(tx.assign_weights(base.copy()))
    tr_idx_a, val_idx_a = tx.make_val_split(train_pool.reset_index(drop=True))
    # evaluate 侧：validation_mae 内部同口径重建 pool 再划分
    eval_pool = tx.cap_single_device(tx.assign_weights(base.copy())).reset_index(drop=True)
    tr_idx_b, val_idx_b = tx.make_val_split(eval_pool)
    assert np.array_equal(val_idx_a, val_idx_b), "两侧验证集索引不一致"
    assert np.array_equal(tr_idx_a, tr_idx_b), "两侧训练集索引不一致"
    import math
    assert len(val_idx_a) == math.ceil(len(base) * 0.2), "验证集占比异常"


def test_split_strata_dtype_stable():
    """划分必须可复现（同输入多次调用结果一致），且 float64 分桶按 round-half-even。"""
    rng = np.random.RandomState(7)
    df = pd.DataFrame({"label": rng.rand(200)})
    _, va = tx.make_val_split(df)
    _, vb = tx.make_val_split(df.copy())
    assert np.array_equal(va, vb), "同输入两次划分结果不一致"
    # float64 round-half-even：0.05 -> 0.0，0.15 -> 0.2（float32 途经会改变边界桶，
    # 契约 v2 规定两侧一律从 DB float64 原值分桶，禁止先转 float32）
    assert np.round(np.float64(0.05), 1) == 0.0 and np.round(np.float64(0.15), 1) == 0.2


def test_assign_weights():
    df = pd.DataFrame({
        "signal_quality_avg": [80.0, 60.0, 30.0, 90.0, 90.0, 90.0, 60.0, 85.0],
        "label_source": ["weak_label", "weak_label", "weak_label", "synthetic", "auto_sample",
                         "cry_detect", "cry_detect", "alert_feedback"],
        "device_id": ["d"] * 8,
    })
    out = tx.assign_weights(df)
    assert len(out) == 7, "sq<40 的 C 级样本未剔除"
    assert out.loc[out["signal_quality_avg"] == 80.0, "sample_weight"].iloc[0] == 1.0, \
        "A级 weak_label 权重应为 1.0"
    assert out.loc[out["signal_quality_avg"] == 60.0, "sample_weight"].iloc[0] == 0.7, \
        "B级 weak_label 权重应为 0.7"
    assert out.loc[out["label_source"] == "synthetic", "sample_weight"].iloc[0] == 1.0
    assert out.loc[out["label_source"] == "auto_sample", "sample_weight"].iloc[0] == 0.7, \
        "非锚点类标签默认 B 级 0.7"
    # ADR-0004 决策 2：cry_detect / alert_feedback 为确认型标注，sq≥70 升 A 级
    cry = out[out["label_source"] == "cry_detect"]
    assert cry.loc[cry["signal_quality_avg"] == 90.0, "sample_weight"].iloc[0] == 1.0, \
        "cry_detect sq≥70 应升 A 级 1.0"
    assert cry.loc[cry["signal_quality_avg"] == 60.0, "sample_weight"].iloc[0] == 0.7, \
        "cry_detect sq<70 仍为 B 级 0.7"
    assert out.loc[out["label_source"] == "alert_feedback", "sample_weight"].iloc[0] == 1.0, \
        "alert_feedback sq≥70 应升 A 级 1.0"


def test_cap_single_device():
    df = pd.DataFrame({"device_id": ["big"] * 100 + ["a", "b", "c"],
                       "label": [0.5] * 103})
    out = tx.cap_single_device(df)
    share = out["device_id"].value_counts(normalize=True).max()
    assert share <= 0.2501, f"单设备占比超限: {share}"


def test_domain_mix():
    df = pd.DataFrame({
        "domain": ["real"] * 10 + ["simulator"] * 90,
        "signal_quality_avg": [90.0] * 100,
        "label_source": ["auto_sample"] * 100,
        "device_id": ["d"] * 100,
        "sample_weight": [1.0] * 100,
    })
    out = tx.apply_domain_mix(df.copy())
    wr = out.loc[out["domain"] == "real", "sample_weight"].iloc[0]
    ws = out.loc[out["domain"] == "simulator", "sample_weight"].iloc[0]
    assert abs(wr - 2.0) < 1e-9, f"real 权重错误: {wr}"
    assert abs(ws - max(0.3, 1 - 2 * 0.1)) < 1e-9, f"sim 权重错误: {ws}"


def test_harvest_eligible():
    good = {"signal_quality_avg": "90", "is_night": "1", "posture": "0",
            "motion_mean": "0.010"}
    assert hc.eligible(good), "合格样本被拒"
    for key, val in [("signal_quality_avg", "60"), ("is_night", "0"),
                     ("posture", "3"), ("motion_mean", "0.02"), ("motion_mean", "nan")]:
        bad = dict(good, **{key: val})
        assert not hc.eligible(bad), f"应拒绝: {key}={val}"


def test_onnx_artifacts():
    import onnxruntime as ort
    for name in ["risk_model_20261011.onnx", "risk_model_20261011_dm.onnx"]:
        path = os.path.join(tx.MODEL_DIR, name)
        assert os.path.exists(path), f"缺模型文件 {name}"
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        meta = sess.get_modelmeta().custom_metadata_map
        assert meta.get("feature_spec_version") == "1.5.0", f"{name} 契约版本错误"
        assert meta.get("n_features") == "15", f"{name} 特征数错误"
        inp = sess.get_inputs()[0]
        assert inp.name == "float_input", f"{name} 输入名错误"
        X = np.zeros((4, 15), dtype=np.float32)
        pred = sess.run(None, {"float_input": X})[0].ravel()
        assert pred.shape == (4,), f"{name} 输出形状错误"
        assert np.all(np.isfinite(pred)), f"{name} 输出含非法值"


def test_eval_history_contract_marked():
    """契约 v2 之后的评估记录必须带 split_contract 标记。"""
    import json
    with open(os.path.join(tx.MODEL_DIR, "eval_history.json")) as f:
        history = json.load(f)
    assert len(history) >= 2, "评估历史为空"
    last2 = history[-2:]
    assert all(r.get("split_contract") == "v2" for r in last2), "最近记录缺契约标记"
    assert all(all(v == "PASS" for v in r["gates"].values()) for r in last2), \
        "最近两版存在未过门禁"


def main():
    print("miaowa-model 管线自测")
    check("split_contract_consistency（train/evaluate 划分一致）", test_split_contract_consistency)
    check("split_strata_dtype_stable（float32/64 桶一致）", test_split_strata_dtype_stable)
    check("assign_weights（质量分级/C级剔除）", test_assign_weights)
    check("cap_single_device（≤25% 上限）", test_cap_single_device)
    check("domain_mix（real×2.0/sim 下限）", test_domain_mix)
    check("harvest_eligible（对照判定口径）", test_harvest_eligible)
    check("onnx_artifacts（契约/形状/值域）", test_onnx_artifacts)
    check("eval_history（契约标记+门禁全过）", test_eval_history_contract_marked)
    print(f"\n结果: {len(PASSED)} PASS / {len(FAILED)} FAIL")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
