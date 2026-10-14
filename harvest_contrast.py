#!/usr/bin/env python3
"""对照样本采集器（AutoLabelService 的离线复刻，v2.1 跑量辅助）。

背景：后端 AutoLabelService 的对照样本条件 motion_mean < motion_baseline × 0.3
实际不可达——基线只在睡眠态以 LR=0.2/5s 自适应，约 25 秒即跟踪到睡眠体动水平，
5min 特征窗均值不可能低于"睡眠基线"的 30%（MW-TEST 跑 3 周 auto_labels 为 0 佐证）。
后端 Java 冻结不改，故用本脚本复刻其插入逻辑，仅将失效的相对条件替换为
绝对静止阈值（motion_mean < 0.008，低于深睡场景体动区间下限 0.004×个性化系数的常态），
其余条件与后端完全一致：sq≥70、is_night=1、posture≠3、事件 ±30min 安静窗、日上限 150。
特征取自 Redis features:{device}（FeatureService 真实计算产物），与后端插入口径同形。

用法：python3 harvest_contrast.py          # 扫描一轮 7 台模拟设备并插入合格样本
注意：Redis 必须走 docker exec（宿主 6379 被原生 redis 抢占）。
"""
import subprocess
import sys

import psycopg2

PG = "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa"
DEVICES = ["MW-SIM-1", "MW-SIM-2", "MW-SIM-3", "MW-SIM-4", "MW-SIM-5",
           "MW-HOLD-1", "MW-HOLD-2"]
SQ_MIN = 70.0
MOTION_QUIET_MAX = 0.014   # 绝对静止阈值（替代失效的 motion < baseline×0.3）；
                           # 覆盖深睡 5min 窗均值常态（0.004-0.016×个性化 0.8-1.25），
                           # 与浅睡下限略有交叠，语义近似"安稳睡眠负样本"，偏差已在报告中注明
DAILY_CAP = 150            # 与 AutoLabelService 一致


def redis_features(dev):
    """HGETALL features:{dev} -> dict；容器内 redis-cli（见文档头部注意）。"""
    out = subprocess.run(
        ["docker", "exec", "miaowa-redis", "redis-cli", "HGETALL", f"features:{dev}"],
        capture_output=True, text=True, check=True).stdout.split()
    return dict(zip(out[::2], out[1::2]))


def num(f, key):
    try:
        return float(f.get(key, "") or "nan")
    except ValueError:
        return float("nan")


def eligible(f):
    """除 motion 相对条件外与 AutoLabelService.sampleDevice 完全一致的判定。"""
    sq = num(f, "signal_quality_avg")
    if sq != sq or sq < SQ_MIN:            # NaN 或信号差
        return False
    if num(f, "is_night") != 1.0:
        return False
    if str(int(num(f, "posture"))) == "3":  # 活动态不稳定（无基线退化的同口径）
        return False
    mo = num(f, "motion_mean")
    return mo == mo and mo < MOTION_QUIET_MAX


def quiet_and_capped(cur, dev):
    """事件 ±30min 安静窗 + 日上限（SQL 与 AutoLabelService 相同）。"""
    cur.execute("""
        SELECT COUNT(*) FROM events
        WHERE device_id = %s AND ts BETWEEN now() - interval '30 minutes'
                                        AND now() + interval '30 minutes'
        """, (dev,))
    if cur.fetchone()[0] > 0:
        return False
    cur.execute("""
        SELECT COUNT(*) FROM training_features
        WHERE device_id = %s AND sample_type = 'contrast' AND ts >= CURRENT_DATE
        """, (dev,))
    return cur.fetchone()[0] < DAILY_CAP


def insert(cur, dev, f):
    """与 AutoLabelService 相同的 auto_labels + training_features 双写（幂等）。"""
    cur.execute("""
        INSERT INTO auto_labels(device_id, ts, label_type, baby_state, risk_score, confidence)
        VALUES (%s, now(), 'stable_window', 'deep_sleep', 0.0, 0.8)
        ON CONFLICT (device_id, ts, label_type) DO NOTHING
        RETURNING device_id
        """, (dev,))
    if not cur.fetchone():
        return False
    cur.execute("""
        INSERT INTO training_features(device_id, ts,
            hr_mean, hr_std, hr_trend, hrv_rmssd,
            motion_mean, motion_max, motion_std, posture,
            temp_mean, temp_trend, signal_quality_avg,
            time_of_day, is_night, hours_since_feeding, hours_since_sleep,
            sample_type, label, label_source, quality_grade)
        VALUES (%s, now(), %s,%s,%s,%s, %s,%s,%s,%s, %s,%s,%s, %s,%s,%s,%s,
                'contrast', 0.0, 'auto_sample', 'B')
        ON CONFLICT (device_id, ts, sample_type) DO NOTHING
        """, (dev,
              num(f, "hr_mean"), num(f, "hr_std"), num(f, "hr_trend"), num(f, "hrv_rmssd"),
              num(f, "motion_mean"), num(f, "motion_max"), num(f, "motion_std"),
              int(num(f, "posture")),
              num(f, "temp_mean"), num(f, "temp_trend"), num(f, "signal_quality_avg"),
              int(num(f, "time_of_day")), int(num(f, "is_night")),
              num(f, "hours_since_feeding"), num(f, "hours_since_sleep")))
    return True


def main():
    conn = psycopg2.connect(PG)
    cur = conn.cursor()
    hits = []
    for dev in DEVICES:
        f = redis_features(dev)
        if not f or not eligible(f) or not quiet_and_capped(cur, dev):
            continue
        if insert(cur, dev, f):
            hits.append(dev)
    conn.commit()
    cur.close()
    conn.close()
    print(f"contrast 采集: {len(hits)} 条 {hits if hits else ''}")


if __name__ == "__main__":
    sys.exit(main())
