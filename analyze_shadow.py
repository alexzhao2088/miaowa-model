#!/usr/bin/env python3
"""
Shadow 阶段离线统计：model_shadow_log 方向一致率 + 偏差分析 + 灰度达标判定
  - 方向一致率 = agree_flag 比例（sign(model-0.5)=sign(rule-0.5)），达标线 ≥70%
  - 偏差：model_score - rule_score 的均值/分位数
  - 按设备/模型版本分层，辅助定位问题设备
  - 退出码 0 = 达标（可切低权重阶段），非零 = 不达标

用法：
    python analyze_shadow.py --pg "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa" [--hours 24]
"""
import argparse
import sys

import psycopg2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pg", required=True)
    ap.add_argument("--hours", type=float, default=24, help="统计最近 N 小时")
    ap.add_argument("--target", type=float, default=0.70, help="方向一致率达标线")
    args = ap.parse_args()

    conn = psycopg2.connect(args.pg)
    cur = conn.cursor()
    cur.execute("""
        SELECT COUNT(*),
               AVG(CASE WHEN agree_flag THEN 1.0 ELSE 0.0 END),
               AVG(model_score - rule_score),
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(model_score - rule_score)),
               PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ABS(model_score - rule_score))
        FROM model_shadow_log
        WHERE ts > now() - (%s || ' hours')::interval
        """, (str(args.hours),))
    n, agree, bias, p50, p95 = cur.fetchone()
    if not n:
        print("model_shadow_log 无数据（确认灰度 phase=shadow 且模型已加载）")
        sys.exit(1)

    print(f"样本量: {n}（最近 {args.hours}h）")
    print(f"方向一致率: {agree:.2%}（达标线 ≥ {args.target:.0%}）")
    print(f"平均偏差(model-rule): {bias:+.4f}  |偏差| P50={p50:.4f} P95={p95:.4f}")

    cur.execute("""
        SELECT device_id, COUNT(*), AVG(CASE WHEN agree_flag THEN 1.0 ELSE 0.0 END)
        FROM model_shadow_log
        WHERE ts > now() - (%s || ' hours')::interval
        GROUP BY device_id ORDER BY 3
        """, (str(args.hours),))
    print("\n按设备:")
    for dev, cnt, a in cur.fetchall():
        flag = "" if a >= args.target else "  <- 低于达标线"
        print(f"  {dev:<12} n={cnt:<6} 一致率={a:.2%}{flag}")

    cur.execute("""
        SELECT model_version, COUNT(*), AVG(CASE WHEN agree_flag THEN 1.0 ELSE 0.0 END)
        FROM model_shadow_log
        WHERE ts > now() - (%s || ' hours')::interval
        GROUP BY model_version
        """, (str(args.hours),))
    print("\n按模型版本:")
    for ver, cnt, a in cur.fetchall():
        print(f"  {ver:<28} n={cnt:<6} 一致率={a:.2%}")

    ok = agree >= args.target
    print(f"\n达标判定: {'PASS（可切低权重阶段 model:phase=low）' if ok else 'FAIL（继续观察或排查）'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
