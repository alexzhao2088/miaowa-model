#!/usr/bin/env python3
"""Phase C 触发值守（PLAN.md Phase S 补充机制）。

触发判定（cold-start-sop 第 0 节）：非 simulator_devices.txt 设备产生
label_source ∈ {weak_label, auto_sample, cry_detect, alert_feedback} 的 training_features
→ 真实数据入场，Phase C 冷启动执行期开始。

防误报口径：① 排除模拟器清单与测试前缀（MW-E2E/MW-TEST/MW-REG）；
② 只统计近 14 天内的样本（历史联调残留不触发——值守 cron 每日跑，新鲜度窗口足够）。

用法：
    python3 phase_c_watch.py            # 检查一次，输出结论；发现真实样本退出码 42
    crontab：17 8 * * * cd .../miaowa-model && python3 phase_c_watch.py >> logs/phase_c_watch.log 2>&1

默认查本地 PG；查生产库需先建立 SSH 隧道：
    ssh -N -L 15432:<生产PG宿主端口映射> miaowa-ecs   # 然后 PG_DSN 指向 127.0.0.1:15432
环境变量 PG_DSN 覆盖默认连接串。
"""
import json
import os
import sys
from pathlib import Path

import psycopg2

DEFAULT_DSN = "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa"
WATCH_LABEL_SOURCES = ("weak_label", "auto_sample", "cry_detect", "alert_feedback")
# 测试设备前缀：e2e/联调设备不属于真实家庭数据，不触发 Phase C
TEST_PREFIXES = ("MW-E2E", "MW-TEST", "MW-REG")


def main():
    dsn = os.environ.get("PG_DSN", DEFAULT_DSN)
    sim_devices = set()
    sim_file = Path(__file__).parent / "simulator_devices.txt"
    if sim_file.exists():
        sim_devices = {l.strip() for l in sim_file.read_text().splitlines()
                       if l.strip() and not l.startswith("#")}

    conn = psycopg2.connect(dsn, connect_timeout=5)
    cur = conn.cursor()
    cur.execute("""
        SELECT device_id, label_source, count(*), max(ts)
        FROM training_features
        WHERE label_source = ANY(%s)
          AND ts >= now() - interval '14 days'
        GROUP BY device_id, label_source
        ORDER BY max(ts) DESC
        """, (list(WATCH_LABEL_SOURCES),))
    rows = cur.fetchall()
    conn.close()

    real = [r for r in rows
            if r[0] not in sim_devices and not r[0].startswith(TEST_PREFIXES)]
    skipped = len(rows) - len(real)
    if not real:
        print(f"[PhaseC] 无真实设备确认型标注样本（扫 {len(rows)} 组，模拟器/测试 {skipped} 组），继续待命")
        return

    print(f"[PhaseC] *** 触发：发现 {len(real)} 组真实设备确认型标注样本 ***")
    for dev, src, n, max_ts in real:
        print(f"  device={dev} source={src} n={n} latest={max_ts}")
    print("[PhaseC] 按 docs/cold-start-sop.md 第 0 节启动 Phase C 冷启动执行期")
    sys.exit(42)


if __name__ == "__main__":
    main()
