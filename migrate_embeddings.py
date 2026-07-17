# ============================================================
# Module: One-shot Embedding Migration (migrate_embeddings.py)
# 模块：一次性向量迁移（更换嵌入模型后全库重嵌）
#
# Triggered at server boot via env var, runs in background thread.
# 由环境变量在启动时触发，后台线程执行，不阻塞服务。
#
# 用法：
#   1. Zeabur Variables 配好新的 OMBRE_EMBEDDING_*（模型/地址/密钥）
#   2. 加 OMBRE_REEMBED_ON_BOOT=true，Save → Redeploy
#   3. 启动后自动：
#      a) 旧 embeddings.db 改名封存为 embeddings.backup-<时间戳>.db（只改名，不删除）
#      b) 用新模型为全部桶（含归档）重新生成向量
#      c) 全部成功后写标记文件 .reembed_done_<模型名>，此后重启不再重复执行
#   4. 开关留着不管也没事（有标记就跳过）；若有失败项则不写标记，下次重启自动续跑
#      （续跑时已完成的桶会跳过，只补失败的）
#
# 桶本体（.md 文件）全程只读，一个字节不改。
# ============================================================

import os
import re
import asyncio
import logging
import threading
from datetime import datetime

logger = logging.getLogger("ombre_brain.migrate")


def _marker_path(buckets_dir: str, model: str) -> str:
    """Marker file path, unique per model / 每个模型一个完成标记。"""
    safe = re.sub(r"[^\w.-]", "_", model or "unknown")
    return os.path.join(buckets_dir, f".reembed_done_{safe}")


async def _run(config: dict) -> None:
    from bucket_manager import BucketManager
    from embedding_engine import EmbeddingEngine

    buckets_dir = config["buckets_dir"]
    model = (config.get("embedding", {}) or {}).get("model", "")
    marker = _marker_path(buckets_dir, model)

    if os.path.exists(marker):
        logger.info(f"Re-embed already done for [{model}], skip / 该模型已迁移过，跳过")
        return

    # --- 1) Shelve old vector db (rename, never delete) — first run only ---
    # --- 封存旧向量库：只改名，不删除；且每个模型只封存一次，续跑不再动 ---
    started = marker + ".started"
    db = os.path.join(buckets_dir, "embeddings.db")
    if not os.path.exists(started):
        if os.path.exists(db):
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup = os.path.join(buckets_dir, f"embeddings.backup-{stamp}.db")
            os.rename(db, backup)
            logger.info(f"Old vectors shelved / 旧向量库已封存: {backup}")
        with open(started, "w", encoding="utf-8") as f:
            f.write(f"{model} @ {datetime.now().isoformat()}\n")
    else:
        logger.info("Resuming previous migration / 检测到未完成的迁移，续跑补漏")

    # --- 2) Fresh engine (recreates empty db) + re-embed all buckets ---
    # --- 新引擎（自动重建空库），全库重嵌 ---
    engine = EmbeddingEngine(config)
    if not engine.enabled:
        logger.error("Embedding engine disabled (no API key?) — abort / 嵌入引擎未启用（缺密钥？），中止迁移")
        return

    mgr = BucketManager(config)
    buckets = await mgr.list_all(include_archive=True)
    total, ok, fail, skip = len(buckets), 0, 0, 0
    logger.info(f"Re-embedding {total} buckets with [{model}] / 开始全库重嵌，共 {total} 桶")

    for i, b in enumerate(buckets, 1):
        content = (b.get("content") or "").strip()
        if not content:
            skip += 1
            continue
        try:
            # Resume support: skip buckets already embedded (fresh db → nothing skipped)
            # 断点续跑：已有向量的桶跳过（首轮是空库，不会跳过任何桶）
            if await engine.get_embedding(b["id"]) is not None:
                skip += 1
                continue
            if await engine.generate_and_store(b["id"], content):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            logger.warning(f"Embed failed for {b['id']}: {e}")
        if i % 20 == 0:
            logger.info(f"Progress / 进度: {i}/{total} (ok={ok} fail={fail})")
            await asyncio.sleep(1)  # gentle pacing / 温和限速

    logger.info(
        f"Re-embed finished / 迁移完成: ok={ok} fail={fail} skip={skip} total={total}"
    )

    # --- 3) Write marker only if nothing failed ---
    # --- 全部成功才写标记；有失败则下次启动续跑 ---
    if fail == 0:
        with open(marker, "w", encoding="utf-8") as f:
            f.write(f"{model} @ {datetime.now().isoformat()}\n")
        logger.info(f"Marker written / 已写完成标记: {marker}")
    else:
        logger.warning(
            "Failures present — marker NOT written; will resume on next boot / "
            "存在失败项，未写标记，下次重启将自动续跑"
        )


def install_export_route(mcp, require_auth, config):
    """
    Register GET /api/export — download the entire buckets dir as .tar.gz.
    注册 /api/export：把整个记忆目录（桶+向量库+标记）打包下载。需先登录仪表盘。
    用法：浏览器登录 /dashboard 后，访问 /api/export 即自动下载 ob-backup-<时间戳>.tar.gz
    """
    import io
    import tarfile
    from starlette.responses import Response, JSONResponse

    buckets_dir = config["buckets_dir"]

    @mcp.custom_route("/api/export", methods=["GET"])
    async def api_export(request):
        err = require_auth(request)
        if err:
            return err
        try:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                tar.add(buckets_dir, arcname="buckets")
            data = buf.getvalue()
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            logger.info(f"Export served / 备份导出完成: {len(data)} bytes")
            return Response(
                content=data,
                media_type="application/gzip",
                headers={
                    "Content-Disposition": f'attachment; filename="ob-backup-{stamp}.tar.gz"'
                },
            )
        except Exception as e:
            logger.error(f"Export failed / 导出失败: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)

    logger.info("Export route installed at /api/export / 备份导出口已就位")


def maybe_reembed_on_boot(config: dict):
    """
    Call from server.py entry point. Env-gated, idempotent, non-blocking.
    在 server.py 启动段调用。环境变量控制、幂等、后台执行不阻塞启动。
    """
    if os.getenv("OMBRE_REEMBED_ON_BOOT", "").lower() not in ("1", "true", "yes", "on"):
        return None

    def _worker():
        try:
            asyncio.run(_run(config))
        except Exception as e:
            logger.error(f"Re-embed crashed (server unaffected) / 迁移线程异常（不影响服务）: {e}")

    t = threading.Thread(target=_worker, name="reembed-on-boot", daemon=True)
    t.start()
    logger.info("Re-embed thread started in background / 迁移已在后台启动")
    return t
