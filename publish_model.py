#!/usr/bin/env python3
"""
模型发布 / 回滚（与 OTA 固件分发同构）
发布：ONNX 传 MinIO bucket=models，对象名 risk_model_YYYYMMDD.onnx；
     读取模型内嵌 feature_spec_version，更新 Redis 指针 model:current / model:feature_spec。
     后端按指针在启动时下载加载，切换需重启后端容器。
回滚：--rollback 将 model:current 回退到上一个版本（按对象名日期排序），并打印提示重启。

用法：
    python publish_model.py --file risk_model_20260905.onnx
    python publish_model.py --rollback
"""
import argparse
import subprocess
import sys

import onnx
from minio import Minio

BUCKET = "models"
DEFAULT_MINIO = "127.0.0.1:9000"
DEFAULT_MINIO_USER = "miaowa"
DEFAULT_MINIO_PASS = "miaowa_dev_2026"
# 注意：本机 127.0.0.1:6379 被宿主 Redis 进程抢占（Docker 端口绑定被遮蔽），
# 必须走 docker exec 操作 miaowa-redis 容器实例。
REDIS_CONTAINER = "miaowa-redis"


def redis_cli(*args: str) -> str:
    out = subprocess.run(["docker", "exec", REDIS_CONTAINER, "redis-cli", *args],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def list_versions(mc: Minio) -> list[str]:
    objs = mc.list_objects(BUCKET, prefix="risk_model_")
    return sorted(o.object_name for o in objs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="待发布的 ONNX 文件")
    ap.add_argument("--rollback", action="store_true", help="回退到上一版本")
    ap.add_argument("--minio", default=DEFAULT_MINIO)
    ap.add_argument("--minio-user", default=DEFAULT_MINIO_USER)
    ap.add_argument("--minio-pass", default=DEFAULT_MINIO_PASS)
    args = ap.parse_args()

    if not args.file and not args.rollback:
        ap.error("必须指定 --file 或 --rollback")

    mc = Minio(args.minio, access_key=args.minio_user, secret_key=args.minio_pass, secure=False)
    if not mc.bucket_exists(BUCKET):
        mc.make_bucket(BUCKET)

    if args.rollback:
        current = redis_cli("GET", "model:current")
        versions = list_versions(mc)
        if not current or current not in versions:
            sys.exit(f"当前指针 {current!r} 不在 MinIO 版本列表 {versions} 中，无法回滚")
        idx = versions.index(current)
        if idx == 0:
            sys.exit("已是最早版本，无上一版可回退")
        prev = versions[idx - 1]
        redis_cli("SET", "model:current", prev)
        m = onnx.load_from_string(mc.get_object(BUCKET, prev).read())
        spec = {p.key: p.value for p in m.metadata_props}.get("feature_spec_version", "unknown")
        redis_cli("SET", "model:feature_spec", spec)
        print(f"回滚完成: {current} -> {prev}（feature_spec_version={spec}）")
        print("请重启后端容器使指针生效: docker restart miaowa-backend")
        return

    m = onnx.load(args.file)
    meta = {p.key: p.value for p in m.metadata_props}
    spec = meta.get("feature_spec_version")
    if not spec:
        sys.exit("模型缺少 feature_spec_version metadata，拒绝发布")
    version = meta.get("model_version") or ""
    object_name = args.file.split("/")[-1]
    if not object_name.startswith("risk_model_"):
        object_name = f"risk_model_{version or 'undated'}.onnx"

    mc.fput_object(BUCKET, object_name, args.file)
    redis_cli("SET", "model:current", object_name)
    redis_cli("SET", "model:feature_spec", spec)
    print(f"发布完成: {object_name} -> MinIO/{BUCKET}，feature_spec_version={spec}")
    print(f"Redis 指针: model:current={object_name} model:feature_spec={spec}")
    print("请重启后端容器加载新版本: docker restart miaowa-backend")


if __name__ == "__main__":
    main()
