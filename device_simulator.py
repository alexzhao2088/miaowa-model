#!/usr/bin/env python3
"""
妙娃虚拟设备模拟器 v2（ADR-0003）：剧本池 + 设备个性化 + 异常场景 + 模拟家长打点
- 剧本池：3 类作息（regular 规律型 / night_waking 夜醒频繁型 / fragmented_naps 小睡碎片化型），
  --seed 随机生成 --pool-size 个剧本，8:2 切训练池 / Held-out Script Pool（尾部 20%）
- 设备 -> 剧本分配：MW-HOLD-* 设备跑 held-out 剧本（样本仅供 evaluate.py 抗过拟合评估，
  train_xgboost.py 按 heldout_devices.txt 剔除出训练），其余设备循环分配训练池剧本
- 设备个性化：hr/motion/temp 基线漂移 + 佩戴松动/夜间脱落概率差异（种子可复现，
  保留原有高斯抖动 + 0.3% 尖峰噪声）
- 异常场景：哭闹升级（长时间未干预：cry 时长 x2.2-3.0 + sq 骤降波动 + 无 soothe 跟随）、
  连续夜醒（night_waking 模板内建 wake/cry/soothe 夜间循环）。仅用现有 5 类事件
  （wake/feeding/diaper/cry/soothe）+ sq 波动表达，不引入新事件类型（契约冻结 1.5.0）
- 剧本场景切换点写 golden_events（scenario 列标记 剧本ID-作息类型，ground truth）
- 模拟家长：事件 80% 打点率、70% 带弱标注、0-10min 随机延迟（随压缩比缩放）、30% 双机重复打点
- 契约 2.0.0 适配：dev-code 登录拿 session（30 天 TTL）、建宝宝绑设备、viewer 经邀请码加入家庭；
  事件走 POST /api/v1/event（ts 秒级 + client_event_id 幂等 + 10min 窗口合并）；
  事件类型仅 feeding/diaper/soothe/sleep/wake，cry 场景以 wake + did_cry=true 弱标注表达
- 压缩只缩短场景时长，不扭曲时间轴（时间戳为真实时间，保证后端 5min/45min 特征窗语义）
- 自动 bootstrap EMQX 设备账号 + ACL；devices 表先注册（外键），已存在跳过

用法：
    python device_simulator.py --ids MW-SIM-1,MW-SIM-2,MW-HOLD-1 --day-min 180 \
        --duration-min 1260 --seed 42
    # 冒烟：--ids MW-SIM-S1,MW-HOLD-S1 --day-min 30 --duration-min 30
"""
import argparse
import json
import random
import threading
import time
import urllib.request
import uuid

import paho.mqtt.client as mqtt
import psycopg2

PG = "postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa"
BACKEND = "http://127.0.0.1:8085"
MQTT_HOST, MQTT_PORT = "127.0.0.1", 1883
EMQX_API = "http://127.0.0.1:18083/api/v5"
EMQX_DASH = ("admin", "miaowa_dev_2026")
DEV_CODE = "888888"  # 短信 dev-code 后门（仅测试环境，与 e2e 一致）

# 场景信号基线：场景 -> (hr范围 bpm, motion范围 g, sq范围)；剧本级抖动在 build_script 叠加
BASE_RANGES = {
    "deep_sleep":  ((95, 105),  (0.004, 0.016), (80, 95)),
    "light_sleep": ((103, 113), (0.014, 0.036), (80, 95)),
    "wake":        ((120, 136), (0.090, 0.220), (75, 92)),
    "feeding":     ((108, 120), (0.025, 0.065), (80, 95)),
    "awake":       ((110, 124), (0.045, 0.130), (78, 93)),
    "cry":         ((126, 142), (0.150, 0.320), (70, 90)),
    "soothe":      ((113, 127), (0.035, 0.100), (78, 93)),
    "diaper":      ((110, 122), (0.045, 0.110), (78, 93)),
}
# 哭闹升级（长时间未干预）：hr/motion 更高、sq 骤降波动表达信号受压
ESCALATED_CRY = ((130, 146), (0.180, 0.350), (50, 68))
EVENT_SCENARIOS = {"wake", "feeding", "diaper", "cry", "soothe"}
SLEEP_SCENES = ("deep_sleep", "light_sleep")
HELDOUT_PREFIX = "MW-HOLD"


# ---------- 剧本池（ADR-0003 决策 2：3 类作息 + 随机种子 + 8:2 切分） ----------

def _tpl_regular(rng):
    """规律型：夜间长睡 + 白天 1 次小睡，事件稀疏。"""
    return [
        ("deep_sleep",  rng.uniform(0.16, 0.22)),
        ("light_sleep", rng.uniform(0.10, 0.16)),
        ("wake",        rng.uniform(0.015, 0.030)),
        ("feeding",     rng.uniform(0.040, 0.060)),
        ("awake",       rng.uniform(0.100, 0.150)),
        ("cry",         rng.uniform(0.015, 0.030)),
        ("soothe",      rng.uniform(0.040, 0.080)),
        ("light_sleep", rng.uniform(0.080, 0.130)),   # 白天小睡
        ("awake",       rng.uniform(0.040, 0.080)),
        ("diaper",      rng.uniform(0.015, 0.030)),
        ("feeding",     rng.uniform(0.030, 0.050)),
        ("deep_sleep",  rng.uniform(0.180, 0.260)),   # 夜间长睡（跨日环接首部深睡）
    ]


def _tpl_night_waking(rng):
    """夜醒频繁型：夜间睡眠被 2-4 轮 wake/cry/soothe 循环打断（连续夜醒异常内建）。"""
    segs = [("deep_sleep", rng.uniform(0.08, 0.12))]
    for _ in range(rng.randint(2, 4)):
        segs.append(("wake", rng.uniform(0.008, 0.015)))
        if rng.random() < 0.7:
            segs.append(("cry", rng.uniform(0.010, 0.025)))
        segs.append(("soothe", rng.uniform(0.015, 0.035)))
        segs.append(("light_sleep", rng.uniform(0.040, 0.080)))
    segs += [
        ("deep_sleep",  rng.uniform(0.08, 0.12)),
        ("wake",        rng.uniform(0.010, 0.020)),
        ("feeding",     rng.uniform(0.040, 0.060)),
        ("awake",       rng.uniform(0.100, 0.140)),
        ("light_sleep", rng.uniform(0.060, 0.100)),
        ("diaper",      rng.uniform(0.015, 0.030)),
        ("awake",       rng.uniform(0.040, 0.080)),
        ("deep_sleep",  rng.uniform(0.100, 0.160)),
    ]
    return segs


def _tpl_fragmented_naps(rng):
    """小睡碎片化型：夜间单段主睡 + 白天 4-6 次碎片小睡。"""
    segs = [("deep_sleep", rng.uniform(0.22, 0.30)),
            ("wake",       rng.uniform(0.010, 0.020)),
            ("feeding",    rng.uniform(0.040, 0.060))]
    for i in range(rng.randint(4, 6)):
        segs.append(("awake", rng.uniform(0.030, 0.060)))
        if i == 1:
            segs.append(("diaper", rng.uniform(0.010, 0.020)))
        if i == 2:
            segs.append(("feeding", rng.uniform(0.030, 0.050)))
        if rng.random() < 0.3:
            segs.append(("cry", rng.uniform(0.010, 0.020)))
            segs.append(("soothe", rng.uniform(0.020, 0.040)))
        segs.append(("light_sleep", rng.uniform(0.030, 0.060)))
    return segs


ARCHETYPES = {
    "regular": _tpl_regular,
    "night_waking": _tpl_night_waking,
    "fragmented_naps": _tpl_fragmented_naps,
}


def build_script(rng, archetype):
    """由作息模板生成一段 24h 剧本：[(场景, 时长占比, hr范围, motion范围, sq范围)]，
    占比归一化为 1；叠加剧本级信号抖动；按概率注入哭闹升级异常。"""
    segs = ARCHETYPES[archetype](rng)
    escalated_idx = set()
    # 异常：哭闹长时间未干预升级（40% 概率）：延长哭闹 + 移除紧随的安抚
    if rng.random() < 0.4:
        cry_idx = [i for i, (n, _) in enumerate(segs) if n == "cry"]
        if cry_idx:
            i = rng.choice(cry_idx)
            segs[i] = ("cry", segs[i][1] * rng.uniform(2.2, 3.0))
            if i + 1 < len(segs) and segs[i + 1][0] == "soothe":
                segs.pop(i + 1)  # 未干预：无安抚事件跟随
            escalated_idx.add(i)
    total = sum(f for _, f in segs)
    hr_jit, mo_jit, sq_jit = rng.uniform(-3, 3), rng.uniform(0.85, 1.20), rng.uniform(-3, 3)
    scenes = []
    for i, (name, frac) in enumerate(segs):
        (h0, h1), (m0, m1), (s0, s1) = ESCALATED_CRY if i in escalated_idx else BASE_RANGES[name]
        scenes.append((name, frac / total,
                       (h0 + hr_jit, h1 + hr_jit),
                       (m0 * mo_jit, m1 * mo_jit),
                       (s0 + sq_jit, s1 + sq_jit)))
    return scenes


def build_script_pool(seed, pool_size):
    """生成剧本池并 8:2 切分：返回 (训练池, Held-out Script Pool)。
    3 类作息轮换分配，同一 seed 完全可复现。"""
    rng = random.Random(seed)
    names = ["regular", "night_waking", "fragmented_naps"]
    pool = []
    for i in range(pool_size):
        arche = names[i % len(names)]
        pool.append({"id": f"s{i + 1:02d}", "archetype": arche,
                     "scenes": build_script(rng, arche)})
    n_held = max(1, round(pool_size * 0.2))
    return pool[:-n_held], pool[-n_held:]


# ---------- 基础设施（EMQX bootstrap / PG / 后端打点） ----------

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


def api_call(method, path, body=None, token=None):
    """后端 API 调用；返回解析后的 JSON（异常时 {"code": -1}）。"""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(f"{BACKEND}{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers=headers)
    try:
        return json.loads(urllib.request.urlopen(req, timeout=10).read())
    except Exception as e:
        return {"code": -1, "error": str(e)}


def login(phone, install_id):
    """验证码登录（dev-code 后门）；未注册手机号自动注册。返回 session_token。"""
    r = api_call("POST", "/api/v1/auth/login",
                 {"phone": phone, "code": DEV_CODE, "app_install_id": install_id})
    token = (r.get("data") or {}).get("session_token")
    if not token:
        raise RuntimeError(f"登录失败 phone={phone}: {r}")
    return token


def setup_accounts(device, idx):
    """契约 2.0.0：为模拟设备准备 owner/viewer 两个家庭账号并绑定设备（幂等）。
    返回 {"owner": token, "viewer": token}。"""
    owner = login(f"1991{idx + 1:07d}", f"sim-owner-{device}")
    viewer = login(f"1992{idx + 1:07d}", f"sim-viewer-{device}")
    bind_code = f"SIM{device[-4:].upper().replace('-', 'X')[:5]}"[:8]

    def bound_device(token):
        r = api_call("GET", "/api/v1/babies", token=token)
        for b in (r.get("data") or {}).get("items", []):
            if b.get("current_device_id") == device:
                return b.get("baby_id") or b.get("id")
        return None

    baby_id = bound_device(owner)
    if baby_id is None:
        r = api_call("POST", "/api/v1/babies",
                     {"name": f"sim-{device}", "gender": "unknown",
                      "birth_date": "2026-03-01", "bind_code": bind_code}, owner)
        baby_id = (r.get("data") or {}).get("baby_id")
        if baby_id is None:  # 可能设备已绑在旧宝宝上被抢占，重查
            baby_id = bound_device(owner)
        if baby_id is None:
            raise RuntimeError(f"宝宝创建/绑定失败 device={device}: {r}")
    if bound_device(viewer) is None:
        code = (api_call("POST", f"/api/v1/babies/{baby_id}/share-code", {}, owner)
                .get("data") or {}).get("code")
        r = api_call("POST", "/api/v1/babies/join", {"code": code}, viewer)
        if r.get("code") != 0:
            print(f"[{device}] viewer 加入家庭失败（双机打点将退化为主账号）: {r}")
    return {"owner": owner, "viewer": viewer}


def post_event(device, token, ts_sec, scenario, with_weak):
    """契约 2.0.0 事件打点：ts 秒级；cry 场景映射为 wake + did_cry=true（API 无 cry 类型）。"""
    body = {"device_id": device, "ts": ts_sec,
            "event_type": "wake" if scenario == "cry" else scenario,
            "client_event_id": uuid.uuid4().hex,
            "did_wake": None, "did_cry": None, "did_intervene": None}
    if with_weak:
        body["did_wake"] = scenario == "wake"
        body["did_cry"] = scenario == "cry"
        body["did_intervene"] = scenario in ("soothe", "diaper")
    r = api_call("POST", "/api/v1/event", body, token)
    if r.get("code") != 0:
        print(f"[{device}] 打点失败 {scenario}: {r}")


class SimDevice(threading.Thread):
    def __init__(self, device_id, args, script, phase, tokens):
        super().__init__(daemon=True)
        self.dev = device_id
        self.args = args
        self.script = script        # {"id", "archetype", "scenes"}
        self.phase = phase          # 0-1，设备间错开剧本相位
        self.tokens = tokens        # {"owner": token, "viewer": token}
        self.rng = random.Random(f"{args.seed}:{device_id}:loop")
        self.stop_at = time.time() + args.duration_min * 60
        # 设备个性化（ADR-0003 决策 2）：种子可复现的个体差异
        prng = random.Random(f"{args.seed}:{device_id}:persona")
        self.hr_offset = prng.uniform(-6, 6)            # hr 基线漂移 bpm
        self.mo_scale = prng.uniform(0.8, 1.25)         # motion 基线漂移
        self.temp_offset = prng.uniform(-0.15, 0.15)    # 体温基线漂移
        self.loosen_p = 0.02 * prng.uniform(0.5, 2.0)   # 佩戴松动概率差异
        self.detach_p = 0.005 * prng.uniform(0.5, 2.0)  # 脱落概率差异
        self.night_detach_p = prng.uniform(0.0, 0.01)   # 夜间额外脱落概率差异

    def scenario_at(self, t):
        """t: 秒（真实时间）。返回 (场景名, hr范围, motion范围, sq范围, 场景起点t0)。"""
        day = self.args.day_min * 60
        pos = ((t / day) + self.phase) % 1.0
        acc = 0.0
        for name, frac, hr, mo, sq in self.script["scenes"]:
            if pos < acc + frac:
                return name, hr, mo, sq, t - (pos - acc) * day
            acc += frac
        name, frac, hr, mo, sq = self.script["scenes"][-1]
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

            # 偶发异常：松动（sq 骤降）/ 脱落（sq 15 + 静止），概率按设备个性化；
            # 睡眠场景叠加夜间脱落差异
            roll = self.rng.random()
            night_detach = name in SLEEP_SCENES and self.rng.random() < self.night_detach_p
            if roll < self.detach_p or night_detach:
                sq, mo, hr = 15, 0.002, self.rng.uniform(*hr_r)
            elif roll < self.detach_p + self.loosen_p:
                sq, mo, hr = self.rng.uniform(38, 55), self.rng.uniform(*mo_r), self.rng.uniform(*hr_r)
            else:
                sq, mo, hr = self.rng.uniform(*sq_r), self.rng.uniform(*mo_r), self.rng.uniform(*hr_r)
            # 设备个性化基线漂移
            hr += self.hr_offset
            mo *= self.mo_scale
            # 噪声：高斯抖动 + 0.3% 尖峰（防剧本过拟合）
            hr += self.rng.gauss(0, 1.5)
            mo = max(0.0, mo + self.rng.gauss(0, 0.004))
            if self.rng.random() < 0.003:
                hr += self.rng.uniform(8, 15)

            pkt = {"device_id": self.dev, "timestamp": int(now),  # 后端按 Unix 秒解析（ofEpochSecond）
                   "ppg_mean": 512.0, "ppg_std": 18.0,
                   "hr_estimated": round(hr, 1), "imu_accel_mean": round(mo, 4),
                   "imu_gyro_mean": 0.001, "temperature": round(36.4 + self.temp_offset + self.rng.gauss(0, 0.08), 2),
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
                post_event(self.dev, self.tokens[m["role"]], m["ts_sec"], m["scenario"], m["weak"])
            time.sleep(5)

        c.loop_stop()
        c.disconnect()
        conn.close()

    def az_for(self, scenario):
        r = self.rng.random()
        if scenario in SLEEP_SCENES:
            if r < 0.75: return -0.9 + self.rng.gauss(0, 0.05)
            if r < 0.90: return self.rng.gauss(0, 0.15)
            return 0.9 + self.rng.gauss(0, 0.05)
        if scenario == "cry":
            return 0.9 + self.rng.gauss(0, 0.08)
        return self.rng.gauss(0.1, 0.3)

    def on_transition(self, conn, new_scenario, seg_start, pending_marks):
        ts_ms = int(seg_start * 1000)
        if new_scenario in EVENT_SCENARIOS:
            # scenario 列标记剧本身份（剧本ID-作息类型），供抗过拟合排查
            insert_golden(conn, self.dev, ts_ms, new_scenario,
                          f"{self.script['id']}-{self.script['archetype']}"[:32])
            if self.rng.random() < 0.8:  # 80% 打点率
                scale = self.args.day_min / 1440.0
                delay = self.rng.uniform(0, 10 * 60) * scale
                owner = {"fire_at": time.time() + delay, "ts_sec": ts_ms // 1000,
                         "scenario": new_scenario, "role": "owner",
                         "weak": self.rng.random() < 0.7}
                pending_marks.append(owner)
                if self.rng.random() < 0.3:  # 30% 双机（viewer 重复打点）
                    viewer = dict(owner)
                    viewer["role"] = "viewer"
                    viewer["fire_at"] = time.time() + delay * self.rng.uniform(0.5, 1.5)
                    pending_marks.append(viewer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", type=int, default=3)
    ap.add_argument("--ids", type=str, default="", help="自定义设备ID（逗号分隔，覆盖 --devices）；"
                    f"{HELDOUT_PREFIX}-* 前缀设备分配 held-out 剧本")
    ap.add_argument("--day-min", type=float, default=360, help="压缩后一昼夜的真实分钟数")
    ap.add_argument("--duration-min", type=float, default=360, help="总运行时长（真实分钟）")
    ap.add_argument("--seed", type=int, default=42, help="剧本池 + 个性化随机种子（可复现）")
    ap.add_argument("--pool-size", type=int, default=10, help="剧本池大小（8:2 切训练/held-out）")
    ap.add_argument("--manifest", default="script_pool_manifest.json", help="剧本分配清单输出路径")
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    ids = [s.strip() for s in args.ids.split(",") if s.strip()] or [f"MW-SIM-{i+1}" for i in range(args.devices)]
    train_pool, held_pool = build_script_pool(args.seed, args.pool_size)
    held_ids = [d for d in ids if d.upper().startswith(HELDOUT_PREFIX)]
    train_ids = [d for d in ids if d not in held_ids]
    if held_ids and not held_pool:
        raise SystemExit("held-out 设备存在但剧本池为空，增大 --pool-size")

    assignment = {}
    for i, d in enumerate(train_ids):
        assignment[d] = train_pool[i % len(train_pool)]
    for i, d in enumerate(held_ids):
        assignment[d] = held_pool[i % len(held_pool)]

    manifest = {
        "seed": args.seed, "pool_size": args.pool_size,
        "train_scripts": [s["id"] for s in train_pool],
        "heldout_scripts": [s["id"] for s in held_pool],
        "devices": {d: {"script": s["id"], "archetype": s["archetype"],
                        "role": "heldout" if d in held_ids else "train"}
                    for d, s in assignment.items()},
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"剧本池 seed={args.seed} 训练池={manifest['train_scripts']} held-out={manifest['heldout_scripts']}")
    for d, meta in manifest["devices"].items():
        print(f"  {d:<12} -> {meta['script']} ({meta['archetype']}, {meta['role']})")

    if not args.skip_bootstrap:
        print("bootstrap EMQX 模拟设备账号与 ACL ...")
        emqx_bootstrap(ids)

    # 契约 2.0.0：devices 表（bind_code 非空 + 7 天绑定窗口，唯一键 device_id）；
    # 已存在则刷新绑定码与窗口（仅模拟设备 ID，不触碰真实设备）
    conn = psycopg2.connect(PG)
    cur = conn.cursor()
    for d in ids:
        cur.execute("""
            INSERT INTO devices(device_id, bind_code, fw_version, bind_code_expires_at)
            VALUES (%s, %s, '1.4.0-sim', now() + interval '7 days')
            ON CONFLICT (device_id) DO UPDATE
            SET bind_code = EXCLUDED.bind_code,
                bind_code_expires_at = EXCLUDED.bind_code_expires_at,
                deleted_at = NULL
            """, (d, f"SIM{d[-4:].upper().replace('-', 'X')[:5]}"[:8]))
    conn.commit()
    cur.close()
    conn.close()

    # 契约 2.0.0：事件打点需账号会话 + 设备绑定（owner/viewer 双账号模拟双机）
    print("为模拟设备准备家庭账号与设备绑定（dev-code 登录）...")
    tokens = {}
    for i, d in enumerate(ids):
        tokens[d] = setup_accounts(d, i)
        print(f"  {d} 账号就绪")

    print(f"启动 {len(ids)} 台虚拟设备：一昼夜={args.day_min}min，运行={args.duration_min}min")
    threads = []
    for i, d in enumerate(ids):
        group = held_ids if d in held_ids else train_ids
        threads.append(SimDevice(d, args, assignment[d],
                                 phase=group.index(d) / max(len(group), 1), tokens=tokens[d]))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    conn = psycopg2.connect(PG)
    cur = conn.cursor()
    cur.execute("SELECT device_id, COUNT(*) FROM golden_events WHERE device_id = ANY(%s) GROUP BY 1", (ids,))
    print("golden_events:", dict(cur.fetchall()))
    cur.execute("SELECT device_id, COUNT(*) FROM events WHERE device_id = ANY(%s) GROUP BY 1", (ids,))
    print("events:", dict(cur.fetchall()))
    cur.execute("""SELECT device_id, sample_type, COUNT(*) FROM training_features
                   WHERE device_id = ANY(%s) GROUP BY 1, 2 ORDER BY 1, 2""", (ids,))
    print("training_features:", cur.fetchall())
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
