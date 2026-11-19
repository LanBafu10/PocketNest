#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aholo OpenAPI 客户端（PocketNest v3）。

与旧版的区别（也是旧流程最致命的两处硬伤）：
1. 上传走 **OUS v2**：GET /asset/v1/token → 在 globalDomain 上 multipart 直传
   → GET /ous/api/v2/upload/status（status=5 成功）。旧版 v1 的 c/d 响应已失效。
2. 建 3D 走 **图生 3D** `/lux3d/v1/generate/img-to-3d/task/create`。
   旧版错用了文生 3D `/generate/text-to-3d/task/create` + 一长串文字描述，
   结果几何形状是由文字「想象」出来的，照片只当风格参考——这正是「建模不对」的根因。
3. 裁图与图生 3D 之间增加 **clean-plate**：`/lux3d/v1/generate/multimodal-to-image/task/create`
   把杂乱实景照改成纯白背景单品图，再送去图生 3D。

所有接口都用 .env 里的 AHOLO_API_KEY；Authorization 头不带 Bearer（官方文档要求）。
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import struct
import time
import urllib.error
import urllib.request
import uuid
import zipfile
import subprocess

from PIL import Image

from pipeline_common import clamp, load_env_config

UA = "PocketNest/3.0"
GLTF_CLI = os.environ.get("PN_GLTF_CLI", "gltf-transform")


def optimize_glb(src_path: str, out_path: str, texture_side: int = 512) -> str:
    """把 Aholo 输出的 2K 贴图压到 512，GLB 体积通常从 ~10MB 降到 ~3MB。

    几何/材质/节点都不动，只重采样贴图；CLI 缺失或失败时回退原文件。
    """
    if not os.path.isfile(GLTF_CLI):
        return src_path
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        r = subprocess.run(
            [GLTF_CLI, "resize", src_path, out_path,
             "--width", str(texture_side), "--height", str(texture_side)],
            capture_output=True, text=True, timeout=180,
        )
        if r.returncode == 0 and os.path.isfile(out_path) and os.path.getsize(out_path) > 1024:
            return out_path
    except Exception:
        pass
    return src_path


class AholoError(RuntimeError):
    pass


def _cfg():
    cfg = load_env_config()
    key = (cfg.get("AHOLO_API_KEY") or "").strip()
    base = (cfg.get("AHOLO_BASE_URL") or "https://api.aholo3d.cn").rstrip("/")
    if not key:
        raise AholoError("未配置 AHOLO_API_KEY")
    return key, base


def _request(url: str, data: bytes = None, method: str = "GET", headers: dict = None, timeout: int = 30,
             auth: str = None):
    h = {"User-Agent": UA}
    if auth:
        h["Authorization"] = auth
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")[:400]
        except Exception:
            pass
        raise AholoError(f"Aholo HTTP {e.code} {url.split('?')[0]} {body}") from e
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise AholoError(f"Aholo 返回非 JSON：{raw[:200]!r}") from e


# --------------------------------------------------------------------------- 上传


def upload_image(file_path: str) -> str:
    """上传本地图片到 Aholo OUS，返回公网可读 URL。"""
    key, base = _cfg()
    tok = _request(f"{base}/asset/v1/token", auth=key, timeout=20)
    ous = tok.get("ousToken")
    dom = (tok.get("globalDomain") or "").rstrip("/")
    block = int(tok.get("blockSize") or 5 * 1024 * 1024)
    if not ous or not dom:
        raise AholoError(f"OUS 凭证异常：{str(tok)[:200]}")

    with open(file_path, "rb") as f:
        raw = f.read()
    if len(raw) > block:
        raise AholoError(f"文件 {len(raw)}B 超过 OUS 单文件上限 {block}B，请先压缩图片")

    md5 = hashlib.md5(raw).hexdigest()
    boundary = "----pn" + uuid.uuid4().hex
    fname = os.path.basename(file_path)
    body = b"".join([
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"md5\"\r\n\r\n{md5}\r\n").encode(),
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{fname}\"\r\n"
         f"Content-Type: application/octet-stream\r\n\r\n").encode(),
        raw,
        (f"\r\n--{boundary}--\r\n").encode(),
    ])
    up = _request(
        f"{dom}/ous/api/v2/single/upload",
        data=body,
        method="POST",
        headers={"ous-token-v2": ous, "Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=90,
    )
    if str(up.get("c")) != "0":
        raise AholoError(f"OUS 上传失败：{str(up)[:300]}")

    for _ in range(40):
        time.sleep(0.5)
        st = _request(f"{dom}/ous/api/v2/upload/status", headers={"ous-token-v2": ous}, timeout=20)
        d = st.get("d") or {}
        if d.get("status") == 5 and d.get("url"):
            return d["url"]
        if d.get("status") in (6, 8):
            raise AholoError(f"OUS 上传终态失败：{str(st)[:300]}")
    raise AholoError("OUS 上传状态轮询超时")


# --------------------------------------------------------------------------- 任务


def create_task(kind: str, payload: dict, concurrency_wait_s: int = 60) -> int:
    """创建 Lux3D 异步任务，返回 taskId。kind: img-to-3d / text-to-3d / multimodal-to-image / image-to-four-view"""
    key, base = _cfg()
    endpoint = f"{base}/lux3d/v1/generate/{kind}/task/create"
    t0 = time.time()
    while time.time() - t0 < concurrency_wait_s:
        resp = _request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
            auth=key,
            timeout=30,
        )
        c = str(resp.get("c"))
        if c == "0" and resp.get("d") is not None:
            d = resp["d"]
            return int(d if not isinstance(d, dict) else d.get("taskId"))
        msg = str(resp.get("m") or "")
        if "CONCURRENCY" in c or "并发" in msg or "concurrency" in msg.lower():
            time.sleep(6.0)
            continue
        raise AholoError(f"{kind} 创建失败：{str(resp)[:300]}")
    raise AholoError(f"{kind} 并发排队超时")


def wait_task(task_id: int, limit_s: int = 420, on_status=None) -> dict:
    key, base = _cfg()
    t0 = time.time()
    while time.time() - t0 < limit_s:
        time.sleep(3.0)
        resp = _request(f"{base}/lux3d/v1/generate/task/get?taskid={task_id}", auth=key, timeout=20)
        if str(resp.get("c")) != "0":
            continue
        d = resp.get("d") or {}
        st = d.get("status")
        if on_status:
            try:
                on_status(st)
            except Exception:
                pass
        if st == 3:
            return d
        if st in (4, 6):
            raise AholoError(f"任务失败 status={st}：{str(d.get('outputs'))[:260]}")
    raise AholoError(f"任务 {task_id} 超过 {limit_s}s 超时")


def _download(url: str, out_path: str, timeout: int = 120) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=timeout) as r:
                data = r.read()
            if not data:
                raise AholoError("下载到 0 字节")
            with open(out_path, "wb") as f:
                f.write(data)
            return out_path
        except Exception as e:  # noqa: BLE001 - IncompleteRead / 网络抖动重试
            last = e
            time.sleep(2 + attempt * 2)
    raise AholoError(f"下载失败（重试 3 次）：{last}")


def _output_urls(d: dict) -> list:
    return [str(o.get("content")) for o in (d.get("outputs") or []) if isinstance(o, dict) and str(o.get("content", "")).startswith("http")]


# --------------------------------------------------------------------------- 单品提纯（clean-plate）


def _white_border_ratio(path: str) -> float:
    """检查四周留白比例，用来判断 clean-plate 是否真的换成了白底。"""
    with Image.open(path) as im:
        im = im.convert("RGB")
        W, H = im.size
        px = im.load()
        edge = max(2, int(min(W, H) * 0.04))
        total = 0
        white = 0
        for x in range(0, W, max(1, W // 80)):
            for y in (0, edge, H - 1 - edge):
                r, g, b = px[x, y]
                total += 1
                white += 1 if (r > 232 and g > 232 and b > 232) else 0
        for y in range(0, H, max(1, H // 80)):
            for x in (0, edge, W - 1 - edge):
                r, g, b = px[x, y]
                total += 1
                white += 1 if (r > 232 and g > 232 and b > 232) else 0
        return white / max(1, total)


def _clean_prompt(item: dict) -> str:
    label = str(item.get("label") or "家具").strip()
    appearance = str(item.get("appearance") or item.get("lux3d_prompt") or "").strip()
    ftype = str(item.get("type") or "other")
    extra = {
        "kitchen_counter": "只保留地柜、台面、水槽与灶台，不要吊柜、不要窗户、不要墙砖。",
        "wall_cabinet": "只保留吊柜本体，不要地柜、台面、墙砖与窗户。",
        "bed": "只保留床架、床垫与枕头，不要床单褶皱、衣物、纸箱、床头柜。",
        "sofa": "只保留沙发本体，不要抱枕堆、纸箱、购物袋、茶几。",
        "wardrobe": "只保留衣柜本体，不要衣物、纸箱与墙面装饰。",
        "desk": "只保留书桌与桌面书架，不要椅子、电脑线材与杂物。",
        "fridge": "只保留冰箱本体，不要周围橱柜与墙面。",
        "toilet": "只保留马桶本体，不要墙面、地砖与纸篓。",
        "vanity_sink": "只保留洗手台与台盆，不要镜子、瓷砖与洗漱用品。",
        "bathtub": "只保留浴缸本体，不要墙面与水管。",
    }.get(ftype, "不要墙面、地板、门窗、其他家具与杂物。")
    return (
        f"参考图里只有一件家具主体：{label}。{('外观特征：' + appearance + '。') if appearance else ''}"
        f"请只保留这件家具本身，{extra}"
        "不要添加原图没有的结构、不要改变它的颜色材质与比例。"
        "输出纯白背景、正面平视、主体完整居中、四周留白、无遮挡、光照均匀的单品产品图。"
    )


def clean_plate(crop_path: str, item: dict, out_path: str, on_state=None) -> dict:
    """脏裁图 → Aholo 多模态单图生成 → 干净白底单品图。返回 {ok, path, url, error}。"""
    if on_state:
        on_state("cleaning")
    try:
        img_url = upload_image(crop_path)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"上传失败：{e}"}

    last_err = ""
    for attempt in range(2):
        prompt = _clean_prompt(item)
        if attempt == 1:
            prompt += " 注意：背景必须是纯白色 #FFFFFF，主体占画面约 65%，四边都要留出白边。"
        try:
            tid = create_task("multimodal-to-image", {"img": img_url, "prompt": prompt})
            d = wait_task(tid, limit_s=240)
            urls = _output_urls(d)
            if not urls:
                last_err = "clean-plate 无输出 URL"
                continue
            _download(urls[0], out_path)
            with Image.open(out_path) as im:
                W, H = im.size
            ratio = _white_border_ratio(out_path)
            if W < 300 or H < 300:
                last_err = f"clean-plate 输出过小 {W}x{H}"
                continue
            if ratio < 0.30:
                last_err = f"clean-plate 白底占比过低 {ratio:.2f}"
                continue
            return {"ok": True, "path": out_path, "url": urls[0], "white_ratio": round(ratio, 3),
                    "size": [W, H], "task_id": tid}
        except Exception as e:  # noqa: BLE001
            last_err = str(e)[:240]
    return {"ok": False, "error": last_err or "clean-plate 失败"}


# --------------------------------------------------------------------------- 图生 3D


def image_to_3d(img_url: str, out_glb_path: str, version: str = "G1-Turbo", face_count: int = 85000,
                is_glass: bool = False, on_state=None) -> dict:
    payload = {
        "img": img_url,
        "version": version,
        "outputFormat": ["glb"],
        "faceCount": int(clamp(face_count, 10000, 300000)),
        "enablePbr": True,
        "aiPredictSize": True,
    }
    tid = create_task("img-to-3d", payload)
    if on_state:
        on_state("generating", tid)
    d = wait_task(tid, limit_s=420)
    urls = _output_urls(d)
    glb = next((u for u in urls if u.split("?")[0].lower().endswith(".glb")), None)
    if glb:
        _download(glb, out_glb_path, timeout=180)
        return {"task_id": tid, "path": out_glb_path, "size": os.path.getsize(out_glb_path)}
    zip_url = next((u for u in urls if u.split("?")[0].lower().endswith(".zip")), None)
    if zip_url:
        raw = urllib.request.urlopen(urllib.request.Request(zip_url, headers={"User-Agent": UA}), timeout=180).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            member = next((n for n in zf.namelist() if n.lower().endswith(".glb")), None)
            if not member:
                raise AholoError("ZIP 内没有 .glb")
            os.makedirs(os.path.dirname(out_glb_path), exist_ok=True)
            with open(out_glb_path, "wb") as f:
                f.write(zf.read(member))
        return {"task_id": tid, "path": out_glb_path, "size": os.path.getsize(out_glb_path)}
    raise AholoError(f"未返回 .glb：{urls[:4]}")


# --------------------------------------------------------------------------- GLB 姿态校准


def _calibrate_glb_pose_geometry(glb_path: str, ftype: str, target_size: list) -> dict:
    """纯二进制读取 GLB 顶点/法线，做 OBB 摆正 + 靠背面判定（无头浏览器不可用时的几何兜底）。

    返回 {glb_fix_deg, glb_aligned_size}；失败时返回零值，前端仍可正常摆放。
    """
    default_res = {"glb_fix_deg": 0.0, "glb_aligned_size": [round(float(v), 3) for v in (target_size or [0.8, 0.8, 0.6])]}
    if not glb_path or not os.path.isfile(glb_path):
        return default_res
    try:
        with open(glb_path, "rb") as f:
            hdr = f.read(12)
            if len(hdr) < 12:
                return default_res
            clen, _ = struct.unpack("<II", f.read(8))
            gjson = json.loads(f.read(clen).decode("utf-8"))
            bhdr = f.read(8)
            if len(bhdr) < 8:
                return default_res
            blen, _ = struct.unpack("<II", bhdr)
            buf = f.read(blen)

        pts, norms = [], []
        for m in gjson.get("meshes", []):
            for p in m.get("primitives", []):
                attrs = p.get("attributes", {})
                pos_idx = attrs.get("POSITION")
                nrm_idx = attrs.get("NORMAL")
                if pos_idx is None:
                    continue
                acc = gjson["accessors"][pos_idx]
                bv = gjson["bufferViews"][acc["bufferView"]]
                off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
                st = bv.get("byteStride", 12)
                cnt = acc["count"]
                step = max(1, cnt // 3000)
                n_off, n_st = None, 12
                if nrm_idx is not None:
                    nacc = gjson["accessors"][nrm_idx]
                    nbv = gjson["bufferViews"][nacc["bufferView"]]
                    n_off = nbv.get("byteOffset", 0) + nacc.get("byteOffset", 0)
                    n_st = nbv.get("byteStride", 12)
                for i in range(0, cnt, step):
                    pts.append(struct.unpack_from("<fff", buf, off + i * st))
                    if n_off is not None:
                        norms.append(struct.unpack_from("<fff", buf, n_off + i * n_st))
        if not pts:
            return default_res

        ys = [p[1] for p in pts]
        ymin, ymax = min(ys), max(ys)
        h = max(0.05, ymax - ymin)

        # 只允许 ±8° 的微调去歪；绝不允许把直角家具拧成 30~45°
        axis_types = {"kitchen_counter", "wardrobe", "bookshelf", "fridge", "wall_cabinet",
                      "vanity_sink", "toilet", "bathtub", "washer"}
        best_deg, best_area = 0.0, 1e18
        if ftype not in axis_types:
            for deg in range(-8, 9):
                rad = math.radians(deg)
                c, s = math.cos(rad), math.sin(rad)
                xs = [p[0] * c + p[2] * s for p in pts]
                zs = [-p[0] * s + p[2] * c for p in pts]
                area = (max(xs) - min(xs)) * (max(zs) - min(zs))
                if area < best_area:
                    best_area, best_deg = area, float(deg)
            if abs(best_deg) <= 2.0:
                best_deg = 0.0

        rad = math.radians(best_deg)
        c, s = math.cos(rad), math.sin(rad)
        rpts = [(p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c) for p in pts]

        xs = [p[0] for p in rpts]
        zs = [p[2] for p in rpts]
        sx, sz = max(0.05, max(xs) - min(xs)), max(0.05, max(zs) - min(zs))
        tw, th, td = [float(v) for v in (target_size or [0.8, 0.8, 0.6])]
        ratio = tw / max(0.05, td)
        need_90 = (ratio > 1.30 and sx < sz * 0.88) or (ratio < 0.77 and sx > sz * 1.12)
        if ftype == "kitchen_counter":
            need_90 = False  # L 型橱柜按两条臂平行坐标轴对齐，转 90° 会把臂拧到墙外
        chosen_q = 90 if need_90 else 0
        # 用「高背侧」判正背面：床/沙发/马桶/书桌的高点在后背，高点在 +Z 就说明
        # 模型正面朝 -Z，需要 180°；转 90° 时按高点在 ±X 选 90/270。
        if ftype in ("bed", "sofa", "toilet", "desk", "chair"):
            top = [p for p in rpts if p[1] >= ymin + 0.72 * h]
            if top:
                tz = sum(p[2] for p in top) / len(top)
                tx = sum(p[0] for p in top) / len(top)
                if chosen_q == 0 and tz > 0.06 * sz:
                    chosen_q = 180
                elif chosen_q == 90:
                    chosen_q = 90 if tx > 0 else 270
        fsx, fsz = (sz, sx) if chosen_q in (90, 270) else (sx, sz)
        total = float(chosen_q) if chosen_q else round(best_deg, 1)
        return {"glb_fix_deg": total,
                "glb_aligned_size": [round(fsx, 3), round(h, 3), round(fsz, 3)],
                "pose_method": "right_angle_snap"}
    except Exception:
        return default_res


def calibrate_glb_pose(
    glb_path: str,
    ftype: str,
    target_size: list,
    crop_path: str = None,
    label: str = "",
    debug_dir: str = None,
    force: bool = False,
) -> dict:
    """优先使用 GLB 四向证件照视觉定标器（含 <glb_path>.pose.json 缓存），失败时优雅回退几何兜底。"""
    if glb_path and os.path.isfile(glb_path):
        try:
            import glb_pose_calibrator

            return glb_pose_calibrator.calibrate_glb_pose(
                glb_path=glb_path,
                ftype=ftype,
                target_size=target_size,
                crop_path=crop_path,
                label=label,
                debug_dir=debug_dir,
                force=force,
            )
        except Exception as e:  # noqa: BLE001
            res = _calibrate_glb_pose_geometry(glb_path, ftype, target_size)
            res["pose_fallback_reason"] = str(e)[:200]
            return res
    return _calibrate_glb_pose_geometry(glb_path, ftype, target_size)
