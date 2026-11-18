#!/bin/bash
# S4 模拟器降频回归（PLAN.md Phase S）：每周 1 次 24h 压缩跑量（7 设备）
#   → harvest_contrast → train（--domain-mix）→ evaluate 五项门禁留档。
# 版本记号 v2.2-rN（不上线，N 自动递增）；门禁 FAIL 即退出码 1，留档 eval_history.json。
# 前置：本地全栈在跑（PG 5432 / 后端 8085 / EMQX 1883 / docker miaowa-redis）。
# cron 建议（本机）：0 3 * * 1  cd /path/to/miaowa-model && ./weekly_regression.sh >> logs/weekly_regression.log 2>&1
set -u
cd "$(dirname "$0")"
mkdir -p logs

PG="postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa"
DEVICES="MW-SIM-1,MW-SIM-2,MW-SIM-3,MW-SIM-4,MW-SIM-5,MW-HOLD-1,MW-HOLD-2"

log() { echo "[$(date '+%F %T')] $*"; }

# ---- 前置检查：本地栈不在跑就跳过（周报留 SKIPPED 记录，不算 FAIL） ----
pg_ready() {
  python3 -c "
import psycopg2, sys
try:
    psycopg2.connect('postgresql://miaowa:miaowa_dev_2026@127.0.0.1:5432/miaowa',
                     connect_timeout=3).close()
except Exception:
    sys.exit(1)
"
}
if ! pg_ready \
   || ! curl -sf -o /dev/null http://127.0.0.1:8085/api/v1/health 2>/dev/null \
   || ! docker exec miaowa-redis redis-cli PING 2>/dev/null | grep -q PONG; then
  log "SKIPPED: 本地栈未就绪（PG/后端/Redis），本周回归跳过"
  exit 0
fi

# ---- 回归序号：从 eval_history.json 找最大 rN ----
N=$(python3 -c "
import json, re
try:
    hist = json.load(open('eval_history.json'))
except Exception:
    hist = []
ns = [int(m.group(1)) for h in hist
      if (m := re.fullmatch(r'v2\.2-r(\d+)', str(h.get('version', ''))))]
print(max(ns, default=0) + 1)
")
VERSION="v2.2-r${N}"
MODEL="risk_model_${VERSION}.onnx"
log "=== 每周回归 ${VERSION} 开始 ==="

# ---- 24h 压缩跑量（一昼夜压到 180 分钟；seed 随周变化，剧本池可复现） ----
SEED=$((1000 + N))
log "跑量：7 设备 × 1 压缩昼夜（180 分钟），seed=${SEED}"
if ! python3 device_simulator.py --ids "$DEVICES" --day-min 180 --duration-min 180 --seed "$SEED" \
        >> logs/sim_r${N}.log 2>&1; then
  log "FAIL: 模拟器跑量异常退出，见 logs/sim_r${N}.log"
  exit 1
fi

# ---- 对照样本采集（离线复刻 AutoLabelService，绝对静止阈值口径） ----
log "对照样本采集"
python3 harvest_contrast.py >> logs/harvest_r${N}.log 2>&1 || log "WARN: harvest 非零退出（继续，样本可能偏少）"

# ---- 训练（域混合加权；不上线，仅留档） ----
log "训练 ${MODEL}"
if ! python3 train_xgboost.py --pg "$PG" --domain-mix \
        --sim-list simulator_devices.txt --heldout-list heldout_devices.txt \
        --out "$MODEL" >> logs/train_r${N}.log 2>&1; then
  log "FAIL: 训练异常退出，见 logs/train_r${N}.log"
  exit 1
fi

# ---- 五项门禁评估（留档 eval_history.json） ----
log "门禁评估 ${VERSION}"
if python3 evaluate.py --pg "$PG" --model "$MODEL" --version "$VERSION" \
        --sim-list simulator_devices.txt --heldout-list heldout_devices.txt \
        --history eval_history.json; then
  log "=== ${VERSION} PASS ==="
else
  log "=== ${VERSION} FAIL（见上方门禁明细，需排查）==="
  exit 1
fi
