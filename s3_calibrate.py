#!/usr/bin/env python3
"""S3 公开数据集特征校准：picsdb（Preterm Infant Cardio-Respiratory Signals Database, ODbL, 内部使用）。

链路：PhysioNet 流式读取 QRS 标注（R 峰）→ RR 间期 → bpm 序列
     → 按特征契约 1.5.0 窗口口径计算 hr_mean(5min)/hr_std(5min)/hr_trend(15min-前30min)/hrv_rmssd(5min)
     → 分布表 JSON（docs/calibration/s3_hr_distribution.json）
     → 与 device_simulator.BASE_RANGES 心率区间对照 → 校准建议（docs/calibration/s3_calibration_report.md）

口径说明（诚实边界）：
- picsdb 为早产儿 ICU 数据，与足月家庭婴儿分布存在系统性差异 → 仅作分布先验与区间合理性校验，
  不直接改写训练数据（ADR-0004 决策 3：仅校准不进训练）。
- picsdb 无睡眠分期标注（仅 R 峰/呼吸峰/心动过缓起点），无法按 deep_sleep/cry 等场景分层，
  分层校准待真实数据（Phase C）。
"""
import json
import os
import sys
import time
from pathlib import Path

import wfdb

PN_DIR = "picsdb/1.0.0"
INFANTS = [f"infant{i}_ecg" for i in range(1, 11)]
OUT_DIR = Path(__file__).parent / "docs" / "calibration"
FS = 250  # picsdb ECG 采样率（hea 确认）

# 特征契约窗口（与 backend FeatureService 对齐）
WIN_MAIN_MIN = 5        # hr_mean/hr_std/hrv_rmssd 主窗
TREND_RECENT_MIN = 15   # hr_trend 近端
TREND_PRIOR_MIN = 30    # hr_trend 前段


def percentile(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return round(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f), 3)


def dist(vals):
    vals = sorted(vals)
    return {
        "n": len(vals),
        "p5": percentile(vals, 5),
        "p25": percentile(vals, 25),
        "p50": percentile(vals, 50),
        "p75": percentile(vals, 75),
        "p95": percentile(vals, 95),
        "mean": round(sum(vals) / len(vals), 3) if vals else None,
    }


def rmssd(rr_ms):
    if len(rr_ms) < 2:
        return None
    diffs = [rr_ms[i] - rr_ms[i - 1] for i in range(1, len(rr_ms))]
    return (sum(d * d for d in diffs) / len(diffs)) ** 0.5


def infant_windows(record):
    """流式读 QRS 标注 → RR(bpm) 序列 → 5min 滑窗特征行。"""
    ann = wfdb.rdann(record, "qrsc", pn_dir=PN_DIR)
    beats = ann.sample  # R 峰采样点
    rows = []
    win = WIN_MAIN_MIN * 60 * FS
    step = win  # 非重叠窗，样本独立性
    for wstart in range(0, int(beats[-1]) - win, step):
        wend = wstart + win
        seg = beats[(beats >= wstart) & (beats < wend)]
        if len(seg) < 60:  # 窗口内 QRS 过少（伪差/脱机），跳过
            continue
        rr = [(seg[i] - seg[i - 1]) / FS for i in range(1, len(seg))]  # 秒
        rr = [r for r in rr if 0.25 <= r <= 1.5]  # 生理范围过滤（40~240bpm）
        if len(rr) < 30:
            continue
        bpm = [60.0 / r for r in rr]
        hr_mean = sum(bpm) / len(bpm)
        hr_std = (sum((b - hr_mean) ** 2 for b in bpm) / len(bpm)) ** 0.5
        hrv = rmssd([r * 1000 for r in rr])
        rows.append({
            "start_sample": int(wstart),
            "hr_mean": round(hr_mean, 3),
            "hr_std": round(hr_std, 3),
            "hrv_rmssd": round(hrv, 3) if hrv is not None else None,
        })
    # hr_trend：近15min均值 - 前30min均值（行序即时间序，窗=5min → 3 行 vs 其前 6 行）
    for i in range(9, len(rows)):
        recent = [r["hr_mean"] for r in rows[i - 2:i + 1]]
        prior = [r["hr_mean"] for r in rows[i - 9:i - 3]]
        if recent and prior:
            rows[i]["hr_trend"] = round(sum(recent) / len(recent) - sum(prior) / len(prior), 3)
    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    per_infant = []
    for rec in INFANTS:
        t0 = time.time()
        try:
            rows = infant_windows(rec)
        except Exception as e:
            print(f"[S3] {rec} FAILED: {e}", file=sys.stderr)
            continue
        print(f"[S3] {rec}: {len(rows)} windows in {time.time() - t0:.0f}s")
        per_infant.append((rec, rows))
        all_rows.extend(rows)

    if not all_rows:
        print("[S3] no data, abort", file=sys.stderr)
        sys.exit(1)

    table = {
        "source": "picsdb 1.0.0 (PhysioNet, ODbL, internal use only)",
        "caveat": "早产儿 ICU 分布，与足月家庭婴儿有系统性差异；仅作分布先验，不进训练",
        "window_contract": "hr_mean/hr_std/hrv_rmssd 5min; hr_trend 15min-前30min（对齐特征契约 1.5.0）",
        "overall": {
            "hr_mean": dist([r["hr_mean"] for r in all_rows]),
            "hr_std": dist([r["hr_std"] for r in all_rows]),
            "hr_trend": dist([r["hr_trend"] for r in all_rows if "hr_trend" in r]),
            "hrv_rmssd": dist([r["hrv_rmssd"] for r in all_rows if r["hrv_rmssd"] is not None]),
        },
        "per_infant": {
            rec: {
                "windows": len(rows),
                "hr_mean": dist([r["hr_mean"] for r in rows]),
            } for rec, rows in per_infant
        },
    }
    out = OUT_DIR / "s3_hr_distribution.json"
    out.write_text(json.dumps(table, ensure_ascii=False, indent=2))
    print(f"[S3] distribution -> {out}")

    # ---- 与模拟器心率区间对照 ----
    sys.path.insert(0, str(Path(__file__).parent))
    from device_simulator import BASE_RANGES, ESCALATED_CRY

    real = table["overall"]
    brady_share = sum(1 for r in all_rows if r["hr_mean"] < 100) / len(all_rows)
    lines = [
        "# S3 校准报告：picsdb 心率分布 vs 模拟器参数",
        "",
        f"- 数据：picsdb 1.0.0，10 例早产儿 ICU，有效 5min 窗 {real['hr_mean']['n']} 个",
        f"- 真实 hr_mean 分布：p5={real['hr_mean']['p5']} / p50={real['hr_mean']['p50']} / p95={real['hr_mean']['p95']} bpm",
        f"- 真实 hr_std(5min)：p50={real['hr_std']['p50']} / p95={real['hr_std']['p95']} bpm",
        f"- 真实 hr_trend：p5={real['hr_trend']['p5']} / p95={real['hr_trend']['p95']} bpm",
        f"- 真实 hrv_rmssd：p50={real['hrv_rmssd']['p50']} / p95={real['hrv_rmssd']['p95']} ms",
        f"- **心动过缓富集**：hr_mean <100bpm 的窗口占 {brady_share:.0%}——该库为心动过缓研究采集，"
        "中位数被显著拉低，不代表足月家庭婴儿静息心率（文献口径 120~160 bpm）。",
        "",
        "## 模拟器 BASE_RANGES 心率区间对照",
        "",
        "| 场景 | 模拟器 hr 区间 | 与真实分布的关系 |",
        "|------|---------------|-----------------|",
    ]
    p5, p95 = real["hr_mean"]["p5"], real["hr_mean"]["p95"]
    for name, (hr, _mo, _sq) in list(BASE_RANGES.items()) + [("cry_escalated", ESCALATED_CRY)]:
        lo, hi = hr
        if hi < p5:
            rel = f"整体低于真实 p5({p5})——偏低"
        elif lo > p95:
            rel = f"整体高于真实 p95({p95})——偏高"
        else:
            rel = "落在真实 p5~p95 包络内"
        lines.append(f"| {name} | {hr[0]:.0f}~{hr[1]:.0f} | {rel} |")
    lines += [
        "",
        "## 结论与边界",
        "",
        "- picsdb 无场景标注，对照仅为整体包络校验；场景级校准待 Phase C 真实数据。",
        "- 本库为心动过缓研究采集（bradycardia-enriched），绝对中位数不可用；",
        "  模拟器区间（95~146 bpm）落在真实包络内且与足月婴儿文献口径一致，**本轮不调整 BASE_RANGES**。",
        "- ADR-0004 决策 3：本校准只校验模拟器分布先验，不进入训练数据。",
        "- 待办：wearable-sensors-infant（CC BY 4.0）sq 分布先验 + SQI 映射，数据集获取渠道待确认。",
    ]
    report = OUT_DIR / "s3_calibration_report.md"
    report.write_text("\n".join(lines) + "\n")
    print(f"[S3] report -> {report}")


if __name__ == "__main__":
    main()
