#!/usr/bin/env python3
"""
妙娃风险评分模型训练脚本（特征契约 v1.4.0，15 维）
数据：training_features（锚点 1.0 / 对照 0.0 / 中间 0.3-0.5，配比 1:1.5:0.5）
规则：
  - 质量分级：A 级=人工锚点(label_source=weak_label)且 sq≥70 -> 权重 1.0；
              B 级=sq≥40 的其余样本（含自动标签）-> 权重 0.7；C 级=sq<40 -> 剔除
  - 单设备样本占比 ≤ 25%（超出随机下采样）
  - 黄金集 label_source='golden' 只评估不训练（剔除出训练池）
  - stratify 按 label 一位小数分桶；固定超参 n=100/depth=4/lr=0.1/subsample=0.8
  - MAE 门禁默认 0.30；导出前 onnxruntime 与 booster 交叉校验（max|Δ| < 1e-5）
输出：risk_model_YYYYMMDD.onnx（输入 float_input [None,15]，metadata 含 feature_spec_version）

用法：
    pip install -r requirements.txt   # 需 brew libomp（XGBoost 依赖）
    python train_xgboost.py --pg "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa"
"""
import argparse
import datetime
import sys

import numpy as np
import pandas as pd

# 特征契约 v1.4.0（顺序即 ONNX 输入顺序，改动必须升 FEATURE_SPEC_VERSION 并重训）
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


def load_data(pg_url: str) -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(pg_url)
    df = pd.read_sql("""
        SELECT device_id, hr_mean, hr_std, hr_trend, hrv_rmssd,
               motion_mean, motion_max, motion_std, posture,
               temp_mean, temp_trend, signal_quality_avg,
               time_of_day, is_night, hours_since_feeding, hours_since_sleep,
               label, sample_type, label_source
        FROM training_features
        WHERE label IS NOT NULL
        """, engine)
    if df.empty:
        sys.exit("training_features 无数据，请先积累样本（自动对照采样 + 弱标注）")
    df["posture_code"] = df["posture"].apply(
        lambda v: POSTURE_MAP.get(v) if isinstance(v, str) else v)  # None/未映射保留 NaN
    return df


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
    """单设备样本占比 ≤25%，超出随机下采样（固定种子保证可复现）。"""
    if df["device_id"].nunique() <= 1:
        return df  # 单设备（模拟期）无需下采样
    parts = []
    cap = int(np.floor(len(df) * MAX_DEVICE_SHARE))
    for dev, g in df.groupby("device_id"):
        parts.append(g.sample(n=min(len(g), cap), random_state=42) if len(g) > cap else g)
    return pd.concat(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg", required=True, help="PostgreSQL 连接串")
    ap.add_argument("--out", default=None, help="默认 risk_model_YYYYMMDD.onnx")
    ap.add_argument("--mae-target", type=float, default=0.30)
    args = ap.parse_args()

    version = datetime.date.today().strftime("%Y%m%d")
    out = args.out or f"risk_model_{version}.onnx"

    df = load_data(args.pg)
    print(f"原始样本: {len(df)}  类型分布:\n{df['sample_type'].value_counts()}")

    golden = df[df["label_source"] == "golden"]
    train_pool = df[df["label_source"] != "golden"].copy()
    if len(golden):
        print(f"黄金集: {len(golden)} 条（仅评估，不进训练）")

    train_pool = assign_weights(train_pool)
    train_pool = cap_single_device(train_pool)
    share = train_pool["device_id"].value_counts(normalize=True)
    print(f"训练池: {len(train_pool)}  设备占比: {dict(share.round(3))}")

    X = train_pool[FEATURE_COLS].astype(np.float32).to_numpy()  # hrv_rmssd NaN 原生支持
    y = train_pool["label"].astype(np.float32).to_numpy()
    w = train_pool["sample_weight"].astype(np.float32).to_numpy()

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error
    from xgboost import XGBRegressor

    strata = np.round(y, 1)  # 连续标签按 0.1 桶分层
    X_tr, X_val, y_tr, y_val, w_tr, _ = train_test_split(
        X, y, w, test_size=0.2, random_state=42, stratify=strata)
    model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8,
        base_score=0.0,  # 与 ONNX 导出 base_values=[0.0] 对齐，保证两侧数值一致
        reg_lambda=1.0, random_state=42,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    mae = mean_absolute_error(y_val, model.predict(X_val))
    print(f"验证集 MAE: {mae:.4f}（目标 < {args.mae_target}）")
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
