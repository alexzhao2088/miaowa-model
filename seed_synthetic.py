#!/usr/bin/env python3
"""合成训练样本生成（验证训练管线 + ONNX 推理闭环用，真实数据到位后废弃）
比例：anchor:contrast:transition = 1:1.5:0.5；label 口径 1.0/0.0/0.3-0.5；
特征列按契约 v1.5.0（is_night/hours_since_*，posture 数字编码见 ADR-0002）；
另生成少量 label_source='golden' 高分段样本供 evaluate.py 黄金集验证。
"""
import random
import psycopg2

random.seed(42)
conn = psycopg2.connect("postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa")
cur = conn.cursor()

DEVICE = "MW-TEST"
N_ANCHOR, N_CONTRAST, N_TRANSITION, N_GOLDEN = 267, 400, 133, 40


def sample(label, motion, hr, sq=None):
    sq = sq if sq is not None else random.randint(70, 95)  # sq 与 label 解耦，防伪相关
    is_night = random.randint(0, 1)
    return {
        "hr_mean": hr + random.gauss(0, 3),
        "hr_std": abs(random.gauss(2, 1)),
        "hr_trend": (label - 0.1) * 4 + random.gauss(0, 1),
        "hrv_rmssd": abs(random.gauss(3 + label * 8, 1.5)),
        "motion_mean": motion + random.gauss(0, 0.01),
        "motion_std": abs(random.gauss(0.02, 0.01)),
        "motion_max": motion * 1.5 + abs(random.gauss(0, 0.02)),
        # posture 列为 smallint 数字编码（ADR-0002）：0 supine / 1 side / 2 prone / 3 active
        "posture": (3 if motion >= 0.15 else random.choice([0, 1, 2, 0, 1] if label < 0.7 else [2, 1, 0, 2, 1])),
        "temp_mean": 36.4 + random.gauss(0, 0.1),
        "temp_trend": random.gauss(0, 0.02),
        "signal_quality_avg": sq,
        "time_of_day": random.randint(0, 23),
        "is_night": is_night,
        "hours_since_feeding": round(random.uniform(0.5, 12.0), 2),
        "hours_since_sleep": round(random.uniform(0.2, 12.0), 2),
    }


def insert(kind, label_fn, rows, source="synthetic"):
    for i, s in enumerate(rows):
        label = label_fn(i) if callable(label_fn) else label_fn
        cur.execute("""
            INSERT INTO training_features(device_id, ts,
                hr_mean, hr_std, hr_trend, hrv_rmssd,
                motion_mean, motion_std, motion_max, posture,
                temp_mean, temp_trend, signal_quality_avg,
                time_of_day, is_night, hours_since_feeding, hours_since_sleep,
                sample_type, label, label_source)
            VALUES (%s, now() - (%s || ' minutes')::interval,
                %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (device_id, ts, sample_type) DO NOTHING
            """, (DEVICE, str(20000 - i * 7 + hash(kind) % 5),
                  s["hr_mean"], s["hr_std"], s["hr_trend"], s["hrv_rmssd"],
                  s["motion_mean"], s["motion_std"], s["motion_max"], s["posture"],
                  s["temp_mean"], s["temp_trend"], s["signal_quality_avg"],
                  s["time_of_day"], s["is_night"], s["hours_since_feeding"], s["hours_since_sleep"],
                  kind, label, source))


# 幂等重种：清掉本设备旧的合成/黄金样本再插入
cur.execute("DELETE FROM training_features WHERE device_id=%s AND label_source IN ('synthetic','golden')", (DEVICE,))

insert("contrast", 0.0, [sample(0.0, random.uniform(0.005, 0.03), random.uniform(95, 110)) for _ in range(N_CONTRAST)])
insert("anchor", 1.0, [sample(1.0, random.uniform(0.12, 0.30), random.uniform(120, 140)) for _ in range(N_ANCHOR)])
insert("transition", lambda i: round(random.uniform(0.3, 0.5), 2),
       [sample(0.4, random.uniform(0.05, 0.12), random.uniform(110, 120)) for _ in range(N_TRANSITION)])
insert("golden", 1.0, [sample(1.0, random.uniform(0.15, 0.30), random.uniform(125, 140)) for _ in range(N_GOLDEN)],
       source="golden")

conn.commit()
cur.close()
conn.close()
print(f"SEEDED anchor={N_ANCHOR} contrast={N_CONTRAST} transition={N_TRANSITION} golden={N_GOLDEN}")
