#!/usr/bin/env python3
"""
妙娃虚拟设备模拟器（P4）：昼夜节律剧本 + 噪声注入 + 模拟家长打点
- 3-5 台虚拟设备（MW-SIM-1..N）并行，每 5s 经 MQTT 发一包（时间戳为真实时间，
  压缩只缩短场景时长，不扭曲时间轴，保证后端 5min/45min 特征窗语义正确）
- 剧本场景切换点写入 golden_events（ground truth，供黄金集评估）
- 模拟家长：事件 80% 打点率、70% 带弱标注、0-10min 随机延迟（随压缩比缩放）、
  30% 概率第二台手机（viewer）重复打点（触发双机合并链路）
- 自动 bootstrap EMQX 设备账号 + ACL（复用 emqx-auth-setup.sh 的 API 模式）

用法：
    python device_simulator.py --devices 3 --day-min 360 --duration-min 360
    # 冒烟：--devices 2 --day-min 30 --duration-min 30
"""
import argparse
import json
import math
import random
import threading
import time
import urllib.request

import paho.mqtt.client as mqtt
import psycopg2

PG = "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa"
BACKEND = "http://127.0.0.1:8085"
MQTT_HOST, MQTT_PORT = "127.0.0.1", 1883
EMQX_API = "http://127.0.0.1:18083/api/v5"
EMQX_DASH = ("admin", "miaowa_dev_2026")

# 昼夜剧本：一段 24h 的场景序列（名称, 时长占比, hr范围, motion范围, sq范围）
# 事件类场景（wake/cry/feeding/diaper/soothe）起点即打点候选点
SCRIPT = [
    ("deep_sleep",  0.20, (96, 104),  (0.005, 0.015), (80, 95)),
    ("light_sleep", 0.15, (104, 112), (0.015, 0.035), (80, 95)),
    ("wake",        0.02, (122, 134), (0.10, 0.20),   (75, 92)),
    ("feeding",     0.05, (110, 118), (0.03, 0.06),   (80, 95)),
    ("awake",       0.13, (112, 122), (0.05, 0.12),   (78, 93)),
    ("cry",         0.03, (128, 140), (0.16, 0.30),   (72, 90)),
    ("soothe",      0.07, (115, 125), (0.04, 0.09),   (78, 93)),
    ("light_sleep", 0.10, (104, 112), (0.015, 0.035), (80, 95)),
    ("diaper",      0.02, (112, 120), (0.05, 0.10),   (78, 93)),
    ("deep_sleep",  0.23, (96, 104),  (0.005, 0.015), (80, 95)),
]
EVENT_SCENARIOS = {"wake", "feeding", "diaper", "cry", "soothe"}


def emqx_token():
    req = urllib.request.Request(f"{EMQX_API}/login", method="POST",
                                 data=json.dumps({"username": EMQX_DASH[0], "password": EMQX_DASH[1]}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())["token"]


def emqx_bootstrap(device_ids):
    """创建模拟设备 EMQX 账号 + 最小权限 ACL（幂等，冲突忽略）。"""
    token = emqx_token()

    def api(path, method="GET", body=None):
        req = urllib.request.Request(f"{EMQX_API}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {token}",
                                              "Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=10)
        except Exception:
            pass  # 409 已存在等幂等冲突忽略

    auth_id = json.loads(urllib.request.urlopen(urllib.request.Request(
        f"{EMQX_API}/authentication", headers={"Authorization": f"Bearer {token}"}),
        timeout=10).read())[0]["id"]
    for dev in device_ids:
        api(f"/authentication/{auth_id}/users", "POST", {"user_id": dev, "password": f"{dev}_sim_2026"})
        api("/authorization/sources/built_in_database/rules/users", "POST", [{
            "username": dev,
            "rules": [
                {"action": "publish", "permission": "allow", "topic": f"miaowa/{dev}/data"},
                {"action": "publish", "permission": "allow", "topic": f"miaowa/{dev}/status"},
                {"action": "subscribe", "permission": "allow", "topic": f"miaowa/{dev}/cmd"},
            ],
        }])


def insert_golden(conn, device, ts_ms, event_type, scenario):
    cur = conn.cursor()
    cur.execute("INSERT INTO golden_events(device_id, ts, event_type, scenario) VALUES (%s, to_timestamp(%s/1000.0), %s, %s)",
                (device, ts_ms, event_type, scenario))
    conn.commit()
    cur.close()


def post_event(device, ts_ms, event_type, app_instance, with_weak, truth_cry):
    body = {"ts": ts_ms, "event_type": event_type, "app_instance_id": app_instance,
            "did_wake": None, "did_cry": None, "did_intervene": None}
    if with_weak:
        body["did_wake"] = event_type == "wake"
        body["did_cry"] = truth_cry
        body["did_intervene"] = event_type in ("soothe", "diaper")
    req = urllib.request.Request(f"{BACKEND}/api/v1/devices/{device}/events", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[{device}] 打点失败 {event_type}: {e}")


class SimDevice(threading.Thread):
    def __init__(self, device_id, args, phase):
        super().__init__(daemon=True)
        self.dev = device_id
        self.args = args
        self.phase = phase          # 0-1，设备间错开剧本相位
        self.rng = random.Random(device_id)
        self.stop_at = time.time() + args.duration_min * 60

    def scenario_at(self, t):
        """t: 秒（真实时间）。返回 (场景名, hr范围, motion范围, sq范围, 场景起点t0)。"""
        day = self.args.day_min * 60
        pos = ((t / day) + self.phase) % 1.0
        acc = 0.0
        for name, frac, hr, mo, sq in SCRIPT:
            if pos < acc + frac:
                return name, hr, mo, sq, t - (pos - acc) * day
            acc += frac
        name, frac, hr, mo, sq = SCRIPT[-1]
        return name, hr, mo, sq, t

    def run(self):
        conn = psycopg2.connect(PG)
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        c.username_pw_set(self.dev, f"{self.dev}_sim_2026")
        c.connect(MQTT_HOST, MQTT_PORT, 30)
        c.loop_start()

        last_scenario, pending_marks = None, []
        while time.time() < self.stop_at:
            now = time.time()
            name, hr_r, mo_r, sq_r, seg_start = self.scenario_at(now)

            if name != last_scenario:
                if last_scenario is not None:
                    self.on_transition(conn, name, seg_start, pending_marks)
                last_scenario = name

            # 偶发异常：2% 松动（sq 骤降）、0.5% 脱落（sq 15 + 静止）
            roll = self.rng.random()
            if roll < 0.005:
                sq, mo, hr = 15, 0.002, self.rng.uniform(*hr_r)
            elif roll < 0.025:
                sq, mo, hr = self.rng.uniform(38, 55), self.rng.uniform(*mo_r), self.rng.uniform(*hr_r)
            else:
                sq, mo, hr = self.rng.uniform(*sq_r), self.rng.uniform(*mo_r), self.rng.uniform(*hr_r)
            # 噪声：高斯抖动 + 0.3% 尖峰（防剧本过拟合）
            hr += self.rng.gauss(0, 1.5)
            mo = max(0.0, mo + self.rng.gauss(0, 0.004))
            if self.rng.random() < 0.003:
                hr += self.rng.uniform(8, 15)

            pkt = {"device_id": self.dev, "timestamp": int(now * 1000),
                   "ppg_mean": 512.0, "ppg_std": 18.0,
                   "hr_estimated": round(hr, 1), "imu_accel_mean": round(mo, 4),
                   "imu_gyro_mean": 0.001, "temperature": round(36.4 + self.rng.gauss(0, 0.08), 2),
                   "signal_quality": int(sq), "wifi_rssi": -58,
                   "accel_x_mean": round(self.rng.gauss(0, 0.05), 3),
                   "accel_y_mean": round(self.rng.gauss(0, 0.05), 3),
                   "accel_z_mean": round(self.az_for(name), 3),
                   "fw_version": "1.4.0-sim", "battery_pct": 80}
            c.publish(f"miaowa/{self.dev}/data", json.dumps(pkt), qos=1)

            # 到期的延迟打点
            due = [m for m in pending_marks if m["fire_at"] <= now]
            pending_marks = [m for m in pending_marks if m["fire_at"] > now]
            for m in due:
                post_event(self.dev, m["ts_ms"], m["event_type"], m["app"], m["weak"], m["cry"])
            time.sleep(5)

        c.loop_stop()
        c.disconnect()
        conn.close()

    def az_for(self, scenario):
        r = self.rng.random()
        if scenario in ("deep_sleep", "light_sleep"):
            if r < 0.75: return -0.9 + self.rng.gauss(0, 0.05)
            if r < 0.90: return self.rng.gauss(0, 0.15)
            return 0.9 + self.rng.gauss(0, 0.05)
        if scenario == "cry":
            return 0.9 + self.rng.gauss(0, 0.08)
        return self.rng.gauss(0.1, 0.3)

    def on_transition(self, conn, new_scenario, seg_start, pending_marks):
        ts_ms = int(seg_start * 1000)
        if new_scenario in EVENT_SCENARIOS:
            insert_golden(conn, self.dev, ts_ms, new_scenario, f"day{int(self.args.day_min)}min")
            if self.rng.random() < 0.8:  # 80% 打点率
                scale = self.args.day_min / 1440.0
                delay = self.rng.uniform(0, 10 * 60) * scale
                owner = {"fire_at": time.time() + delay, "ts_ms": ts_ms,
                         "event_type": new_scenario, "app": f"sim-owner-{self.dev}",
                         "weak": self.rng.random() < 0.7, "cry": new_scenario == "cry"}
                pending_marks.append(owner)
                if self.rng.random() < 0.3:  # 30% 双机（viewer 重复打点）
                    viewer = dict(owner)
                    viewer["app"] = f"sim-viewer-{self.dev}"
                    viewer["fire_at"] = time.time() + delay * self.rng.uniform(0.5, 1.5)
                    pending_marks.append(viewer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=3)
    ap.add_argument("--day-min", type=float, default=360, help="压缩后一昼夜的真实分钟数")
    ap.add_argument("--duration-min", type=float, default=360, help="总运行时长（真实分钟）")
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    ids = [f"MW-SIM-{i+1}" for i in range(args.devices)]
    if not args.skip_bootstrap:
        print("bootstrap EMQX 模拟设备账号与 ACL ...")
        emqx_bootstrap(ids)

    # sensor_data 等表有 device_id 外键：先登记模拟设备（幂等）
    conn = psycopg2.connect(PG)
    cur = conn.cursor()
    for d in ids:
        cur.execute("""
            INSERT INTO device(device_id, device_token, fw_version)
            VALUES (%s, %s, '1.4.0-sim') ON CONFLICT (device_id) DO NOTHING
            """, (d, f"{d}_sim_2026"))
    conn.commit()
    cur.close()
    conn.close()

    print(f"启动 {len(ids)} 台虚拟设备：一昼夜={args.day_min}min，运行={args.duration_min}min")
    threads = [SimDevice(d, args, phase=i / len(ids)) for i, d in enumerate(ids)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    conn = psycopg2.connect(PG)
    cur = conn.cursor()
    cur.execute("SELECT device_id, COUNT(*) FROM golden_events WHERE device_id = ANY(%s) GROUP BY 1", (ids,))
    print("golden_events:", dict(cur.fetchall()))
    cur.execute("SELECT device_id, COUNT(*) FROM event_record WHERE device_id = ANY(%s) GROUP BY 1", (ids,))
    print("event_record:", dict(cur.fetchall()))
    cur.execute("""SELECT sample_type, COUNT(*) FROM training_features
                   WHERE device_id = ANY(%s) GROUP BY 1""", (ids,))
    print("training_features:", dict(cur.fetchall()))
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
