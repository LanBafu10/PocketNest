#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PocketNest v3 流水线公共层。

职责：
- 环境配置与存储路径（项目 JSON 原子读写）
- 多模态 VLM 网关调用（带模型回退、重试与耗时日志）
- 图片编码 / 户型叠加图渲染
- 流水线阶段日志（写入 project.pipeline，前端与单步调试接口都可读）
- 几何与文本小工具

设计原则：本文件不含任何「小家」专属尺寸或房间名，所有几何均由
户型图像素 + VLM 估计尺寸 + 程序化校验得到。
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = os.environ.get("PN_BASE_DIR", ".")
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
ENV_FILE = os.path.join(BASE_DIR, ".env")

VLM_PROXY_URL = os.environ.get("PN_VLM_URL", "https://api.openai.com/v1/chat/completions")
# 空间拓扑/家具三角定位都用强多模态模型；失败时逐级回退。
# ZHU_MASTER_PLAN：一律默认使用 gemini-4-hydrogen-high（思考链需要 max_tokens=16384）。
FP_VLM_MODEL = os.environ.get("PN_FP_MODEL", "gemini-4-hydrogen-high")
ROOM_VLM_MODEL = os.environ.get("PN_ROOM_MODEL", "gemini-4-hydrogen-high")
AUDIT_VLM_MODEL = os.environ.get("PN_AUDIT_MODEL", "gemini-4-hydrogen-high")

PIPELINE_VERSION = "3.1-door-anchored"

os.makedirs(STORAGE_DIR, exist_ok=True)


# --------------------------------------------------------------------------- 配置


def load_env_config() -> dict:
    cfg = {
        "AHOLO_API_KEY": os.environ.get("AHOLO_API_KEY", ""),
        "AHOLO_BASE_URL": os.environ.get("AHOLO_BASE_URL", "https://api.aholo3d.cn"),
    }
    if os.path.isfile(ENV_FILE):
        with open(ENV_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    return cfg


def get_vlm_api_key() -> str:
    if os.environ.get("VLM_API_KEY"):
        return os.environ["VLM_API_KEY"]
    try:
        with open("./config/models.json", "r", encoding="utf-8") as f:
            return (json.load(f).get("providers", {}).get("agproxy", {}) or {}).get("apiKey", "local-proxy")
    except Exception:
        return "local-proxy"


# --------------------------------------------------------------------------- 项目存储


def project_dir(pid: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "", pid)
    if not safe:
        raise ValueError("invalid project id")
    return os.path.join(STORAGE_DIR, safe)


def load_project(pid: str):
    pfile = os.path.join(project_dir(pid), "project.json")
    if not os.path.isfile(pfile):
        return None
    with open(pfile, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project(pid: str, data: dict):
    pdir = project_dir(pid)
    os.makedirs(pdir, exist_ok=True)
    data["updated_at"] = int(time.time())
    tmp = os.path.join(pdir, "project.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, os.path.join(pdir, "project.json"))


# --------------------------------------------------------------------------- VLM


def encode_image_for_vlm(img_path: str, max_side: int = 1100) -> tuple:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return "image/jpeg", b64


def extract_json_block(text: str) -> dict:
    text = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if m:
        text = m.group(1)
    else:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s : e + 1]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def call_vlm_json(
    prompt: str,
    image_paths: list,
    timeout: int = 180,
    max_images: int = 10,
    model: str = None,
    max_tokens: int = 16384,
) -> dict:
    """调用本地 VLM 网关，返回解析后的 JSON。失败会依次换模型重试。"""
    content = []
    for p in image_paths[:max_images]:
        if p and os.path.isfile(p):
            mtype, b64 = encode_image_for_vlm(p)
            content.append({"type": "image_url", "image_url": {"url": f"data:{mtype};base64,{b64}"}})
    content.append({
        "type": "text",
        "text": prompt + "\n\n注意：字符串值内部切勿使用未转义的英文双引号 \"，只输出合法纯 JSON。",
    })

    chosen = model or ROOM_VLM_MODEL
    chain = []
    for m in (chosen, "gemini-4-hydrogen-high", "gemini-2.5-pro", "claude-sonnet-4.6"):
        if m and m not in chain:
            chain.append(m)

    last_err = None
    for try_model in chain:
        t0 = time.time()
        try:
            payload = {
                "model": try_model,
                "max_tokens": max_tokens,
                "temperature": 0.1,
                "messages": [{"role": "user", "content": content}],
            }
            req = urllib.request.Request(
                VLM_PROXY_URL,
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {get_vlm_api_key()}"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                res = json.loads(resp.read().decode("utf-8"))
            txt = (((res.get("choices") or [{}])[0].get("message") or {}).get("content")) or ""
            out = extract_json_block(txt)
            out["_vlm"] = {"model": try_model, "elapsed_s": round(time.time() - t0, 1)}
            return out
        except Exception as e:  # noqa: BLE001 - 逐级回退，最后抛出
            last_err = e
            # 网关重启/瞬时拒绝连接时给 2s 喘息，再换下一个模型
            if isinstance(e, (urllib.error.URLError, ConnectionError, TimeoutError)):
                time.sleep(2.0)
            continue
    raise RuntimeError(f"VLM 调用失败（已尝试 {chain}）：{last_err}")


# --------------------------------------------------------------------------- 图片


def image_size(path: str) -> tuple:
    with Image.open(path) as im:
        return im.size


def draw_floorplan_overlay(fp_path: str, rooms: list, out_path: str) -> str:
    """把识别出的房间框、编号、房间名画到户型图上，供第二轮 VLM 审核。"""
    with Image.open(fp_path) as im:
        im = im.convert("RGB")
        W, H = im.size
    draw = ImageDraw.Draw(im)
    palette = [
        (230, 57, 70), (29, 53, 87), (42, 157, 143), (233, 196, 106),
        (131, 56, 236), (255, 107, 53), (0, 119, 182), (76, 201, 240),
        (188, 108, 37), (106, 76, 147), (42, 96, 153), (214, 40, 40),
    ]
    font = None
    for fpath in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(fpath):
            try:
                font = ImageFont.truetype(fpath, max(16, int(min(W, H) * 0.028)))
                break
            except Exception:
                font = None
    for idx, r in enumerate(rooms):
        bb = r.get("bbox_1000")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x1, y1, x2, y2 = [float(v) for v in bb]
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        px1, py1 = x1 / 1000.0 * W, y1 / 1000.0 * H
        px2, py2 = x2 / 1000.0 * W, y2 / 1000.0 * H
        color = palette[idx % len(palette)]
        for w in range(4):
            draw.rectangle([px1 - w, py1 - w, px2 + w, py2 + w], outline=color)
        tag = f"{idx + 1}.{r.get('name', '')}"
        try:
            tw = draw.textlength(tag, font=font)
        except Exception:
            tw = len(tag) * 12
        draw.rectangle([px1, py1, px1 + tw + 10, py1 + 26], fill=color)
        draw.text((px1 + 5, py1 + 4), tag, fill=(255, 255, 255), font=font)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    im.save(out_path, format="JPEG", quality=88)
    return out_path


# --------------------------------------------------------------------------- 日志


def log_stage(proj: dict, stage: str, status: str = "ok", detail: str = "", **extra):
    """把阶段日志追加到 project['pipeline']，最多保留 300 条。"""
    pipe = proj.setdefault("pipeline", {})
    pipe.setdefault("version", PIPELINE_VERSION)
    pipe.setdefault("events", [])
    if pipe.get("stage") != stage:
        pipe["stage"] = stage
        pipe["stage_started_at"] = int(time.time())
    ev = {"ts": int(time.time()), "stage": stage, "status": status}
    if detail:
        ev["detail"] = str(detail)[:900]
    for k, v in extra.items():
        if v is not None:
            ev[k] = v
    pipe["events"].append(ev)
    if len(pipe["events"]) > 300:
        del pipe["events"][: len(pipe["events"]) - 300]
    return ev


def pipeline_warning(proj: dict, stage: str, msg: str):
    pipe = proj.setdefault("pipeline", {})
    w = pipe.setdefault("warnings", [])
    w.append({"ts": int(time.time()), "stage": stage, "msg": str(msg)[:400]})
    if len(w) > 100:
        del w[: len(w) - 100]
    log_stage(proj, stage, "warn", msg)


# --------------------------------------------------------------------------- 几何


def rect_overlap(a: list, b: list) -> tuple:
    """a/b = [x1,y1,x2,y2]（米或像素均可）。返回 (重叠宽, 重叠高)。"""
    ox = min(a[2], b[2]) - max(a[0], b[0])
    oz = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, ox), max(0.0, oz)


def median(vals: list, default: float = 0.0) -> float:
    vals = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not vals:
        return default
    vals.sort()
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def norm_label(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()
