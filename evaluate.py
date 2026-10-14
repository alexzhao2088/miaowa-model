#!/usr/bin/env python3
"""
模型评估报告 v2（ADR-0003 决策 5：验收口径换锚，替代 PRD 绝对 MAE 阶梯）
对指定 ONNX 模型输出五项指标，四项门禁全过才允许出模型：
  [1] 绝对 MAE（训练池验证集 20% 留出，与 train_xgboost.py 同种子同 stratify）
      —— 降级为回归监控：不许显著劣于上版（margin 默认 0.05，对照 --history 上一版）
  [2] Held-out 剧本测试（核心新增门禁）：heldout_devices.txt 设备（跑 20% 留出剧本）
      样本 MAE 与训练池剧本 MAE（[1] 验证集口径）差值 < 0.08
  [3] 跨设备留一验证（leave-one-device-out）MAE：报告并逐版不劣化（margin 同 [1]）
  [4] 信号分层鲁棒性：sq 40-70 区间与 sq≥70 区间 MAE 差值 < 0.05（沿用补传分层口径）
  [5] 黄金集高分段命中率：对照 golden_events 表（剧本真相），事件 ±20min 窗口内
      label>0.7 样本中 pred>0.5 的比例 ≥ 60%
评估结束向 --history（默认 eval_history.json）追加一条记录，供逐版回归对比。

用法：
    python evaluate.py --pg "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa" \
        --model risk_model_20260918.onnx --version v2.1
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import pandas as pd

from train_xgboost import (FEATURE_COLS, MODEL_DIR, assign_weights,
                           cap_single_device, load_data, make_val_split,
                           read_device_list)

GOLDEN_WINDOW_MIN = 20  # 黄金事件对照窗口（±min），覆盖锚点 15min 窗 + 打点延迟
SPLIT_CONTRACT = "v2"  # 与 train_xgboost.py 共用 make_val_split；v1 历史记录的
                       # 验证集口径因行序不一致不可比，回归对比只对照同契约版本


def predict(sess, X):
    return sess.run(None, {"float_input": X.astype(np.float32)})[0].ravel()


def validation_mae(df):
    """与 train_xgboost.py 完全同口径的 20% 验证集 MAE（split contract v2）：
    同序数据（load_data ORDER BY）+ 同质量分级/单设备上限 + 共用 make_val_split。"""
    pool = cap_single_device(assign_weights(df.copy())).reset_index(drop=True)
    _, val_idx = make_val_split(pool)
    val = pool.iloc[val_idx]
    return float(val["abs_err"].mean()), len(val)


def lodo_mae(df):
    """跨设备留一验证：逐设备留出，其余设备按训练口径（质量分级加权）训练，
    在留出设备上算 MAE。返回 (逐设备dict, mean)。"""
    from xgboost import XGBRegressor
    pool = assign_weights(df.copy())  # 与训练同口径：C 级剔除、A/B 加权
    devices = sorted(pool["device_id"].unique())
    if len(devices) < 2:
        return {}, None
    per_dev = {}
    for dev in devices:
        tr, te = pool[pool["device_id"] != dev], pool[pool["device_id"] == dev]
        if len(te) < 5 or len(tr) < 50:
            continue
        m = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                         subsample=0.8, base_score=0.0, reg_lambda=1.0,
                         random_state=42)
        m.fit(tr[FEATURE_COLS].astype(np.float32).to_numpy(),
              tr["label"].astype(np.float32).to_numpy(),
              sample_weight=tr["sample_weight"].astype(np.float32).to_numpy())
        pred = m.predict(te[FEATURE_COLS].astype(np.float32).to_numpy())
        per_dev[dev] = float(np.mean(np.abs(pred - te["label"].astype(float).to_numpy())))
    mean = float(np.mean(list(per_dev.values()))) if per_dev else None
    return per_dev, mean


def golden_hit_rate(df, pg_url, devices):
    """对照 golden_events 表：事件 ±20min 窗口内 label>0.7 样本 pred>0.5 的比例。"""
    from sqlalchemy import create_engine
    gold = pd.read_sql("""
        SELECT device_id, ts, event_type FROM golden_events
        WHERE device_id = ANY(%s)
        """, create_engine(pg_url), params=(list(devices),))
    if gold.empty:
        return None, 0
    cand = df[df["label"] > 0.7]
    if cand.empty:
        return None, 0
    win = pd.Timedelta(minutes=GOLDEN_WINDOW_MIN)
    hit_idx = set()
    for dev, g in gold.groupby("device_id"):
        c = cand[cand["device_id"] == dev]
        if c.empty:
            continue
        ts_ser = c["ts"]
        if ts_ser.dt.tz is not None:  # timestamptz -> UTC 朴素时间，与 golden ts 对齐
            ts_ser = ts_ser.dt.tz_convert("UTC").dt.tz_localize(None)
        ts = ts_ser.to_numpy()
        for t in g["ts"]:
            if getattr(t, "tzinfo", None) is not None:
                t = t.tz_convert("UTC").tz_localize(None)
            lo, hi = np.datetime64(t - win), np.datetime64(t + win)
            hit_idx.update(c.index[(ts >= lo) & (ts <= hi)].tolist())
    near = cand.loc[sorted(hit_idx)]
    if near.empty:
        return None, 0
    return float((near["pred"] > 0.5).mean()), len(near)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", default=None, help="版本标记（如 v2.1），写入评估历史")
    ap.add_argument("--sim-list", default=os.path.join(MODEL_DIR, "simulator_devices.txt"))
    ap.add_argument("--heldout-list", default=os.path.join(MODEL_DIR, "heldout_devices.txt"))
    ap.add_argument("--heldout-gap", type=float, default=0.08, help="门禁[2]：held-out 与训练池 MAE 差值上限")
    ap.add_argument("--sq-gap", type=float, default=0.05, help="门禁[4]：sq 分层 MAE 差值上限")
    ap.add_argument("--golden-hit-target", type=float, default=0.60, help="门禁[5]：黄金集高分段命中率下限")
    ap.add_argument("--regression-margin", type=float, default=0.05, help="回归监控容忍劣化幅度")
    ap.add_argument("--history", default=os.path.join(MODEL_DIR, "eval_history.json"))
    args = ap.parse_args()

    df = load_data(args.pg)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    print(f"模型: {args.model}  metadata={sess.get_modelmeta().custom_metadata_map}")

    df["pred"] = predict(sess, df[FEATURE_COLS].astype(np.float32).to_numpy())
    df["abs_err"] = (df["pred"] - df["label"].astype(float)).abs()

    sim_devices = read_device_list(args.sim_list)
    heldout_devices = read_device_list(args.heldout_list)
    df["domain"] = np.where(df["device_id"].isin(sim_devices), "simulator", "real")
    held_mask = df["device_id"].isin(heldout_devices)
    train_df = df[~held_mask & (df["label_source"] != "golden")].copy()
    held_df = df[held_mask & (df["label_source"] != "golden")].copy()
    print(f"样本: 总 {len(df)}（训练池设备 {len(train_df)} / held-out 设备 {len(held_df)}）"
          f"  Domain 分布: {df.groupby('domain')['device_id'].nunique().to_dict()}")

    history = []
    if os.path.exists(args.history):
        with open(args.history) as f:
            history = json.load(f)
    prev = history[-1] if history else None
    gates = {}

    # [1] 绝对 MAE（回归监控，非硬门禁；显著劣于上版才 FAIL）
    mae, n_val = validation_mae(train_df)
    gate1 = "PASS"
    prev_same_contract = prev if prev and prev.get("split_contract") == SPLIT_CONTRACT else None
    if prev_same_contract and prev_same_contract.get("mae") is not None \
            and mae > prev_same_contract["mae"] + args.regression_margin:
        gate1 = "FAIL"
    print(f"\n[1] 验证集 MAE: {mae:.4f}（n={n_val}，上版 {prev_same_contract['mae']:.4f}，回归监控 margin={args.regression_margin}）{gate1}"
          if prev_same_contract and prev_same_contract.get("mae") is not None else
          f"\n[1] 验证集 MAE: {mae:.4f}（n={n_val}，划分契约 {SPLIT_CONTRACT} 基线重置，回归监控不判定）{gate1}"
          if prev and prev.get("split_contract") != SPLIT_CONTRACT else
          f"\n[1] 验证集 MAE: {mae:.4f}（n={n_val}，首版记录基线，回归监控不判定）{gate1}")
    gates["mae_regression"] = gate1

    # [2] Held-out 剧本测试（核心新增）
    if len(held_df):
        mae_held = float(held_df["abs_err"].mean())
        gap = abs(mae_held - mae)
        gate2 = "PASS" if gap < args.heldout_gap else "FAIL"
        print(f"[2] Held-out 剧本测试: held-out MAE={mae_held:.4f}（n={len(held_df)}）"
              f" vs 训练池 MAE={mae:.4f}  差值={gap:.4f}（目标 < {args.heldout_gap}）{gate2}")
    else:
        mae_held, gap, gate2 = None, None, "SKIP"
        print("[2] Held-out 剧本测试: held-out 设备无样本，SKIP（跑量后必须有数据）")
    gates["heldout_script"] = gate2

    # [3] 跨设备留一验证（报告 + 逐版不劣化）
    per_dev, lodo = lodo_mae(train_df)
    gate3 = "PASS"
    if prev and prev.get("lodo_mean") is not None and lodo is not None \
            and lodo > prev["lodo_mean"] + args.regression_margin:
        gate3 = "FAIL"
    if lodo is None:
        gate3 = "SKIP"
        print("[3] 跨设备留一验证: 设备数或样本量不足，SKIP")
    else:
        detail = "  ".join(f"{d}={v:.4f}" for d, v in per_dev.items())
        base = f"上版 {prev['lodo_mean']:.4f}，" if prev and prev.get("lodo_mean") is not None else "首版记录基线，"
        print(f"[3] 跨设备留一验证: mean MAE={lodo:.4f}（{base}逐版不劣化）{gate3}\n    {detail}")
    gates["lodo"] = gate3

    # [4] 信号分层鲁棒性（sq 40-70 vs sq≥70，沿用补传分层口径）
    pool = df[df["label_source"] != "golden"]
    mid = pool[(pool["signal_quality_avg"] >= 40) & (pool["signal_quality_avg"] < 70)]
    high = pool[pool["signal_quality_avg"] >= 70]
    if len(mid) and len(high):
        mae_mid, mae_high = float(mid["abs_err"].mean()), float(high["abs_err"].mean())
        sq_gap = abs(mae_mid - mae_high)
        gate4 = "PASS" if sq_gap < args.sq_gap else "FAIL"
        print(f"[4] 信号分层鲁棒性: sq40-70 MAE={mae_mid:.4f}（n={len(mid)}）"
              f" vs sq≥70 MAE={mae_high:.4f}（n={len(high)}）  差值={sq_gap:.4f}（目标 < {args.sq_gap}）{gate4}")
    else:
        mae_mid, mae_high, sq_gap, gate4 = None, None, None, "SKIP"
        print("[4] 信号分层鲁棒性: sq 分层样本不足，SKIP")
    gates["sq_robust"] = gate4

    # [5] 黄金集高分段命中率（对照 golden_events 剧本真相）
    hit, n_gold = golden_hit_rate(df, args.pg, df["device_id"].unique())
    if hit is None:
        gate5 = "SKIP"
        print("[5] 黄金集高分段命中率: golden_events 窗口内无 label>0.7 样本，SKIP")
    else:
        gate5 = "PASS" if hit >= args.golden_hit_target else "FAIL"
        print(f"[5] 黄金集高分段命中率: {hit:.2%}（n={n_gold}，目标 ≥ {args.golden_hit_target:.0%}，"
              f"对照 golden_events ±{GOLDEN_WINDOW_MIN}min）{gate5}")
    gates["golden_hit"] = gate5

    # 分层明细（参考）
    print("\n[参考] 分层 MAE:")
    for name, row in pool.groupby("sample_type")["abs_err"].agg(["mean", "count"]).iterrows():
        print(f"    sample_type={name:<12} MAE={row['mean']:.4f} n={int(row['count'])}")
    for name, row in pool.groupby("label_source")["abs_err"].agg(["mean", "count"]).iterrows():
        print(f"    label_source={name:<10} MAE={row['mean']:.4f} n={int(row['count'])}")

    record = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "version": args.version, "model": os.path.basename(args.model),
        "mae": mae, "heldout_mae": mae_held, "heldout_gap": gap,
        "lodo_mean": lodo, "lodo_per_device": per_dev,
        "sq_mid_mae": mae_mid, "sq_high_mae": mae_high, "sq_gap": sq_gap,
        "golden_hit": hit, "golden_n": n_gold, "gates": gates,
        "split_contract": SPLIT_CONTRACT,
    }
    history.append(record)
    with open(args.history, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    failed = [k for k, v in gates.items() if v == "FAIL"]
    skipped = [k for k, v in gates.items() if v == "SKIP"]
    print(f"\n评估结论: {'FAIL（' + ','.join(failed) + '）' if failed else 'PASS'}"
          + (f"  跳过项: {','.join(skipped)}" if skipped else ""))
    print(f"评估记录已追加: {args.history}（共 {len(history)} 版）")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
