#!/usr/bin/env python3
"""
妙娃风险评分模型训练脚本（特征契约 v1.5.0，15 维，posture 数字编码见 ADR-0002）
数据：training_features（锚点 1.0 / 对照 0.0 / 中间 0.3-0.5，配比 1:1.5:0.5）
规则：
  - 质量分级：A 级=人工锚点(label_source=weak_label)且 sq≥70 -> 权重 1.0；
              B 级=sq≥40 的其余样本（含自动标签）-> 权重 0.7；C 级=sq<40 -> 剔除
  - 单设备样本占比 ≤ 25%（超出随机下采样）
  - 黄金集 label_source='golden' 只评估不训练（剔除出训练池）
  - Domain 划分（ADR-0003 决策 3）：读取 simulator_devices.txt 清单打 simulator/real 域标，
    零 DB 迁移；v2.1 仅打标与分布报告
  - 域混合加权（ADR-0003 决策 4，v2.2）：--domain-mix 开启后 real 域权重 ×2.0，
    simulator 域权重 = max(0.3, 1 − 2×real 样本占比)；--pseudo-real-device 将 1 台
    模拟器设备按 real 域加权（Pseudo-real Domain 自测加权逻辑与分层报告）
  - Held-out Script Pool（ADR-0003 决策 2）：heldout_devices.txt 设备跑留出剧本，
    其样本剔除出训练池，专供 evaluate.py 抗过拟合评估
  - stratify 按 label 一位小数分桶；固定超参 n=100/depth=4/lr=0.1/subsample=0.8
  - MAE 门禁默认 0.30；导出前 onnxruntime 与 booster 交叉校验（max|Δ| < 1e-5）
输出：risk_model_YYYYMMDD.onnx（输入 float_input [None,15]，metadata 含 feature_spec_version）

用法：
    pip install -r requirements.txt   # 需 brew libomp（XGBoost 依赖）
    python train_xgboost.py --pg "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa"
"""
import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd

# 特征契约 v1.5.0（顺序即 ONNX 输入顺序，改动必须升 FEATURE_SPEC_VERSION 并重训）
FEATURE_COLS = [
    "hr_mean", "hr_std", "hr_trend", "hrv_rmssd",
    "motion_mean", "motion_max", "motion_std", "posture_code",
    "temp_mean", "temp_trend", "signal_quality_avg",
    "time_of_day", "is_night", "hours_since_feeding", "hours_since_sleep",
]
FEATURE_SPEC_VERSION = "1.5.0"
# posture 数字编码（ADR-0002）：DB 列为 smallint，0/1/2/3 直传；兼容历史字符串值
POSTURE_MAP = {"supine": 0, "side": 1, "prone": 2, "active": 3}
MAX_DEVICE_SHARE = 0.25
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


def read_device_list(path: str) -> set:
    """读取设备清单文件（每行一个 device_id，# 注释）；文件不存在返回空集。"""
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {ln.strip() for ln in f if ln.strip() and not ln.startswith("#")}


def load_data(pg_url: str) -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(pg_url)
    df = pd.read_sql("""
        SELECT device_id, ts, hr_mean, hr_std, hr_trend, hrv_rmssd,
               motion_mean, motion_max, motion_std, posture,
               temp_mean, temp_trend, signal_quality_avg,
               time_of_day, is_night, hours_since_feeding, hours_since_sleep,
               label, sample_type, label_source
        FROM training_features
        WHERE label IS NOT NULL
        ORDER BY device_id, ts
        """, engine)
    if df.empty:
        sys.exit("training_features 无数据，请先积累样本（自动对照采样 + 弱标注）")
    df["posture_code"] = df["posture"].apply(
        lambda v: POSTURE_MAP.get(v) if isinstance(v, str) else v)  # None/未映射保留 NaN
    return df


def make_val_split(pool: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """训练/评估共用的验证集划分（split contract v2）。
    要求 pool 行序确定（load_data ORDER BY + cap_single_device 定序），
    标签按 float64 精度 0.1 桶分层；最小桶 <2 条时退化不分层。
    返回 (train_idx, val_idx) 位置索引。"""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(pool))
    strata = np.round(pool["label"].astype(float).to_numpy(), 1)  # 必须 float64，float32 舍入边界不同
    if pd.Series(strata).value_counts().min() < 2:
        print(f"警告: 最小标签桶仅 {pd.Series(strata).value_counts().min()} 条，本次划分不做分层")
        strata = None
    tr_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=seed,
                                       stratify=strata)
    return tr_idx, val_idx


def assign_weights(df: pd.DataFrame) -> pd.DataFrame:
    """质量分级 -> sample_weight；C 级剔除。golden 只评估不训练。"""
    sq = df["signal_quality_avg"].astype(float)
    is_anchor_human = df["label_source"] == "weak_label"
    df["sample_weight"] = np.where(is_anchor_human & (sq >= 70), 1.0, 0.7)
    df.loc[df["label_source"] == "synthetic", "sample_weight"] = 1.0  # 管线验证数据视为 A 级
    kept = df[(sq >= 40) | (df["label_source"] == "synthetic")].copy()
    dropped = len(df) - len(kept)
    if dropped:
        print(f"C 级剔除（sq<40）: {dropped} 条")
    return kept


def cap_single_device(df: pd.DataFrame) -> pd.DataFrame:
    """单设备样本占比 ≤25%（占下采样后全集），超出随机下采样。
    固定种子 + 迭代收敛保证可复现；多数设备同时超限时逐轮收紧。"""
    if df["device_id"].nunique() <= 1:
        return df  # 单设备（模拟期）无需下采样
    out = df
    for _ in range(10):  # 每次下采样改变总量，迭代到占比收敛
        counts = out["device_id"].value_counts()
        share = counts / len(out)
        over = share[share > MAX_DEVICE_SHARE]
        if over.empty:
            return out
        dev = over.index[0]  # 占比最高者优先收紧
        others = len(out) - int(counts[dev])  # 其余设备样本数
        keep = int(np.floor(others * MAX_DEVICE_SHARE / (1 - MAX_DEVICE_SHARE)))
        g = out[out["device_id"] == dev]
        out = pd.concat([out[out["device_id"] != dev],
                         g.sample(n=min(len(g), keep), random_state=42)])
    return out


def apply_domain_mix(df: pd.DataFrame, real_weight: float = 2.0) -> pd.DataFrame:
    """域混合加权（ADR-0003 决策 4）：real ×2.0，sim = max(0.3, 1 − 2×real占比)。
    在质量分级权重的基础上按域乘系数；pseudo_real 域按 real 处理。"""
    real_mask = df["domain"].isin(["real", "pseudo_real"])
    real_share = real_mask.mean()
    sim_weight = max(0.3, 1.0 - 2.0 * real_share)
    df["domain_weight"] = np.where(real_mask, real_weight, sim_weight)
    df["sample_weight"] = df["sample_weight"] * df["domain_weight"]
    print(f"域混合加权: real占比={real_share:.3f}  real权重={real_weight}  "
          f"sim权重={sim_weight:.2f}（下限 0.3）")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg", required=True, help="PostgreSQL 连接串")
    ap.add_argument("--out", default=None, help="默认 risk_model_YYYYMMDD.onnx")
    ap.add_argument("--mae-target", type=float, default=0.30)
    ap.add_argument("--sim-list", default=os.path.join(MODEL_DIR, "simulator_devices.txt"),
                    help="模拟器设备清单（Domain 划分，清单外默认 real）")
    ap.add_argument("--heldout-list", default=os.path.join(MODEL_DIR, "heldout_devices.txt"),
                    help="Held-out 剧本池设备清单（样本剔除出训练，仅供评估）")
    ap.add_argument("--domain-mix", action="store_true",
                    help="v2.2：开启域混合加权（real ×2.0，sim = max(0.3, 1−2×real占比)）")
    ap.add_argument("--pseudo-real-device", default="",
                    help="v2.2 自测：将指定模拟器设备标记为 pseudo_real 域（按 real 加权）")
    args = ap.parse_args()

    version = datetime.date.today().strftime("%Y%m%d")
    out = args.out or f"risk_model_{version}.onnx"

    df = load_data(args.pg)
    print(f"原始样本: {len(df)}  类型分布:\n{df['sample_type'].value_counts()}")

    # Domain 划分（ADR-0003 决策 3）：清单内 simulator，清单外默认 real；v2.1 仅打标报告
    sim_devices = read_device_list(args.sim_list)
    df["domain"] = np.where(df["device_id"].isin(sim_devices), "simulator", "real")
    if args.pseudo_real_device:
        if args.pseudo_real_device not in sim_devices:
            print(f"警告: pseudo-real 设备 {args.pseudo_real_device} 不在 simulator_devices.txt，"
                  f"按 real 处理即可，无需此开关")
        df.loc[df["device_id"] == args.pseudo_real_device, "domain"] = "pseudo_real"
    dom = df.groupby("domain")["device_id"].agg(["count", "nunique"])
    print(f"Domain 分布:\n{dom}")

    # Held-out Script Pool 设备：样本专供抗过拟合评估，剔除出训练
    heldout_devices = read_device_list(args.heldout_list)
    held_mask = df["device_id"].isin(heldout_devices)
    if held_mask.any():
        print(f"Held-out 剧本池设备剔除出训练: {int(held_mask.sum())} 条 "
              f"（{sorted(df.loc[held_mask, 'device_id'].unique())}）")
        df = df[~held_mask].copy()

    golden = df[df["label_source"] == "golden"]
    train_pool = df[df["label_source"] != "golden"].copy()
    if len(golden):
        print(f"黄金集: {len(golden)} 条（仅评估，不进训练）")

    train_pool = assign_weights(train_pool)
    train_pool = cap_single_device(train_pool)
    if args.domain_mix:
        train_pool = apply_domain_mix(train_pool)
    share = train_pool["device_id"].value_counts(normalize=True)
    print(f"训练池: {len(train_pool)}  设备占比: {dict(share.round(3))}")

    X = train_pool[FEATURE_COLS].astype(np.float32).to_numpy()  # hrv_rmssd NaN 原生支持
    y = train_pool["label"].astype(np.float32).to_numpy()
    w = train_pool["sample_weight"].astype(np.float32).to_numpy()
    dom_arr = train_pool["domain"].to_numpy()

    from sklearn.metrics import mean_absolute_error
    from xgboost import XGBRegressor

    # 连续标签按 0.1 桶分层；与 evaluate.py 共用 make_val_split（split contract v2）
    tr_idx, val_idx = make_val_split(train_pool)
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    w_tr = w[tr_idx]
    dom_tr, dom_val = dom_arr[tr_idx], dom_arr[val_idx]
    model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
        base_score=0.0,  # 与 ONNX 导出 base_values=[0.0] 对齐，保证两侧数值一致
        reg_lambda=1.0, random_state=42,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    pred_val = model.predict(X_val)
    mae = mean_absolute_error(y_val, pred_val)
    print(f"验证集 MAE: {mae:.4f}（目标 < {args.mae_target}）")
    # 域分层报告（v2.2 自测：验证加权逻辑下各域表现）
    for d in sorted(set(dom_arr)):
        m = dom_val == d
        if m.any():
            print(f"  [域分层] {d}: n={int(m.sum())}  MAE={mean_absolute_error(y_val[m], pred_val[m]):.4f}")
    if mae >= args.mae_target:
        sys.exit("MAE 未达标，先积累更多样本或调参")

    from onnx_export import booster_to_onnx, verify_against_booster
    onnx_model = booster_to_onnx(model.get_booster(), len(FEATURE_COLS),
                                 FEATURE_SPEC_VERSION, model_version=version)
    max_diff = verify_against_booster(onnx_model, model.get_booster(), X_val[:64])
    print(f"ONNX 交叉校验 max|Δ|: {max_diff:.2e}")
    if max_diff > 1e-5:
        sys.exit("ONNX 与 XGBoost 预测不一致，导出中止")

    with open(out, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX 已导出: {out}（{len(FEATURE_COLS)} 维, feature_spec_version={FEATURE_SPEC_VERSION}）")


if __name__ == "__main__":
    main()
