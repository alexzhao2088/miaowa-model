#!/usr/bin/env python3
"""
模型评估报告：对指定 ONNX 模型在 training_features 上输出
  1. 验证集 MAE（与 train_xgboost.py 同种子同 stratify 切分，口径一致）
  2. 黄金集高分段命中率：label_source='golden' 且 label>0.7 的样本中 pred>0.5 的比例
  3. 分层 MAE：按 sample_type（anchor/contrast/transition）与 label_source 分组
退出码：验证集 MAE 超阈值或黄金命中率 <80% 时非零（供发布前门禁）。

用法：
    python evaluate.py --pg "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa" \
        --model risk_model_20260905.onnx
"""
import argparse
import sys

import numpy as np
import onnxruntime as ort

from train_xgboost import FEATURE_COLS, load_data


def predict(sess, X):
    return sess.run(None, {"float_input": X.astype(np.float32)})[0].ravel()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mae-target", type=float, default=0.30)
    ap.add_argument("--golden-hit-target", type=float, default=0.80)
    args = ap.parse_args()

    df = load_data(args.pg)
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    print(f"模型: {args.model}  metadata={sess.get_modelmeta().custom_metadata_map}")

    X_all = df[FEATURE_COLS].astype(np.float32).to_numpy()
    df["pred"] = predict(sess, X_all)
    df["abs_err"] = (df["pred"] - df["label"].astype(float)).abs()

    train_pool = df[df["label_source"] != "golden"].reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(train_pool))
    _, val_idx = train_test_split(idx, test_size=0.2, random_state=42,
                                  stratify=np.round(train_pool["label"].astype(float), 1))
    val = train_pool.iloc[val_idx]
    mae = float(val["abs_err"].mean())
    print(f"\n[1] 验证集 MAE: {mae:.4f}（目标 < {args.mae_target}，n={len(val)}）")

    golden = df[(df["label_source"] == "golden") & (df["label"] > 0.7)]
    if len(golden):
        hit = float((golden["pred"] > 0.5).mean())
        print(f"[2] 黄金集高分段命中率: {hit:.2%}（目标 ≥ {args.golden_hit_target:.0%}，n={len(golden)}）")
    else:
        hit = None
        print("[2] 黄金集为空（模拟器跑量后由剧本真相补充），跳过命中率")

    print("[3] 分层 MAE:")
    by_type = df.groupby("sample_type")["abs_err"].agg(["mean", "count"])
    for name, row in by_type.iterrows():
        print(f"    sample_type={name:<12} MAE={row['mean']:.4f} n={int(row['count'])}")
    by_src = df.groupby("label_source")["abs_err"].agg(["mean", "count"])
    for name, row in by_src.iterrows():
        print(f"    label_source={name:<10} MAE={row['mean']:.4f} n={int(row['count'])}")

    ok = mae < args.mae_target and (hit is None or hit >= args.golden_hit_target)
    print(f"\n评估结论: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
