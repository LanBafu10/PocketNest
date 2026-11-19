#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PocketNest v3 API 服务（127.0.0.1:18792）。

接口分层（每一层都能单独调用，方便断点排查）：
  POST /api/pocketnest-v2/projects                                   建项目
  POST /api/pocketnest-v2/projects/{pid}/floorplan                  上传户型图并解析（pass1+audit+scale+snap）
  POST /api/pocketnest-v2/projects/{pid}/floorplan/reparse          只重跑户型解析
  GET  /api/pocketnest-v2/projects/{pid}/floorplan/debug            pass1 / audit / 尺度 / 叠加图
  POST /api/pocketnest-v2/projects/{pid}/rooms/{rid}/photos          上传房间照片
  POST /api/pocketnest-v2/projects/{pid}/rooms/{rid}/analyze         单房间解析（pass1+merge+audit+solve）
  POST /api/pocketnest-v2/projects/{pid}/build3d                     全屋编排（可 reparse / force / 只跑一间）
  POST /api/pocketnest-v2/projects/{pid}/pipeline/step               单步调试：floorplan_parse / floorplan_audit / room_analyze / clean_plate / img_to_3d
  GET  /api/pocketnest-v2/projects/{pid}/pipeline                    流水线阶段日志、告警、逐件状态
  POST /api/pocketnest-v2/projects/{pid}/rooms/{rid}/furniture       旋转 / 移动 / 重试 / 补单件
  GET  /api/pocketnest-v2/health                                     健康检查

后台队列：2 个 worker，每件家具按
  脏裁图 → Aholo clean-plate（多模态单图生成，纯白底）→ Aholo 图生 3D（img-to-3d）→ GLB 姿态校准
执行，状态写回 project.json，前端轮询即可看到进度。
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import queue
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import aholo_client as aholo
import pipeline_common as pc
import pipeline_floorplan as pf
import pipeline_room as pr
from pipeline_common import (
    PIPELINE_VERSION,
    load_env_config,
    load_project,
    log_stage,
    project_dir,
    save_project,
)

PROJECT_LOCK = threading.RLock()
LUX3D_QUEUE: "queue.Queue" = queue.Queue()
QUEUED_KEYS: set = set()
WORKERS = int(os.environ.get("PN_LUX_WORKERS", "2"))


# --------------------------------------------------------------------------- 资产生命周期


def find_item(proj: dict, rid: str, iid: str):
    for r in proj.get("rooms", []):
        if r["id"] != rid:
            continue
        for it in (r.get("scene_spec") or {}).get("furniture", []):
            if it.get("id") == iid:
                return r, it
    return None, None


def update_item(pid: str, rid: str, iid: str, **updates):
    with PROJECT_LOCK:
        proj = load_project(pid)
        if not proj:
            return None
        r, it = find_item(proj, rid, iid)
        if not it:
            return None
        it.update(updates)
        save_project(pid, proj)
        return it


def enqueue_jobs(pid: str, custom_key: str = "") -> int:
    proj = load_project(pid)
    if not proj:
        return 0
    count = 0
    for r in proj.get("rooms", []):
        for it in (r.get("scene_spec") or {}).get("furniture", []):
            if it.get("glb_url") or it.get("lux_status") in ("running", "generating", "succeeded"):
                continue
            if not it.get("crop_rel"):
                continue
            qkey = f"{pid}:{r['id']}:{it['id']}"
            if qkey in QUEUED_KEYS:
                continue
            QUEUED_KEYS.add(qkey)
            update_item(pid, r["id"], it["id"], lux_status="queued", lux_error=None)
            LUX3D_QUEUE.put((pid, r["id"], it["id"], custom_key))
            count += 1
    return count


def _set_state(pid, rid, iid, status, **extra):
    update_item(pid, rid, iid, lux_status=status, **extra)


def lux_worker_loop(worker_id: int):
    while True:
        pid, rid, iid, custom_key = LUX3D_QUEUE.get()
        qkey = f"{pid}:{rid}:{iid}"
        try:
            proj = load_project(pid)
            if not proj:
                continue
            room, item = find_item(proj, rid, iid)
            if not item:
                continue
            pdir = project_dir(pid)
            crop_abs = os.path.join(pdir, item["crop_rel"]) if item.get("crop_rel") else None
            if not crop_abs or not os.path.isfile(crop_abs):
                _set_state(pid, rid, iid, "failed", lux_error="裁图不存在")
                continue
            clean_src = os.path.join(pdir, item.get("clean_source_rel") or item["crop_rel"])
            if not os.path.isfile(clean_src):
                clean_src = crop_abs

            # 1) clean-plate：已经有干净图就复用
            clean_rel = item.get("clean_rel")
            clean_abs = os.path.join(pdir, clean_rel) if clean_rel else None
            if not (clean_abs and os.path.isfile(clean_abs)):
                out_rel = f"clean/{iid}.png"
                out_abs = os.path.join(pdir, out_rel)
                res = aholo.clean_plate(clean_src, item, out_abs, on_state=lambda s: _set_state(pid, rid, iid, s))
                if res.get("ok"):
                    clean_rel, clean_abs = out_rel, out_abs
                    update_item(pid, rid, iid, clean_rel=out_rel,
                                clean_url=f"/api/pocketnest-v2/files/{pid}/{out_rel}",
                                clean_error=None, clean_meta={"white_ratio": res.get("white_ratio"), "size": res.get("size")})
                else:
                    update_item(pid, rid, iid, clean_error=res.get("error") or "clean-plate 失败")
                    pc.log_stage(proj, f"lux.clean:{item.get('label')}", "warn",
                                 f"clean-plate 失败，退回原裁图：{str(res.get('error'))[:160]}")
                    clean_abs = None
            # 2) 图生 3D：必须用图生接口，不能用文生
            img_url = None
            if clean_abs and os.path.isfile(clean_abs):
                img_url = aholo.upload_image(clean_abs)
            else:
                img_url = aholo.upload_image(crop_abs)
            _set_state(pid, rid, iid, "uploading")
            version = "G1" if item.get("is_glass") else "G1-Turbo"
            raw_rel = f"models/{iid}.glb"
            raw_abs = os.path.join(pdir, raw_rel)
            res = aholo.image_to_3d(img_url, raw_abs, version=version,
                                    on_state=lambda s, tid=None: _set_state(
                                        pid, rid, iid, s, **({"lux_task_id": tid} if tid else {})))
            # 2.5) 贴图 512 压缩，避免 34 件家具把前端显存打爆
            opt_rel = f"models_opt/{iid}.glb"
            used_abs = aholo.optimize_glb(raw_abs, os.path.join(pdir, opt_rel))
            used_rel = opt_rel if used_abs == os.path.join(pdir, opt_rel) else raw_rel
            pose = aholo.calibrate_glb_pose(
                used_abs,
                item.get("type", "other"),
                item.get("size_m"),
                crop_path=crop_abs,
                label=item.get("label", ""),
                debug_dir=os.path.join(pdir, "debug", "glb_4view"),
            )
            update_item(pid, rid, iid, lux_status="succeeded", lux_task_id=res.get("task_id"),
                        lux_version=version, glb_rel=used_rel,
                        glb_url=f"/api/pocketnest-v2/files/{pid}/{used_rel}",
                        glb_fix_deg=pose["glb_fix_deg"], glb_aligned_size=pose["glb_aligned_size"],
                        lux_error=None, lux_clean_used=bool(clean_abs))
        except Exception as e:  # noqa: BLE001
            update_item(pid, rid, iid, lux_status="failed", lux_error=str(e)[:220])
        finally:
            QUEUED_KEYS.discard(qkey)
            LUX3D_QUEUE.task_done()


# --------------------------------------------------------------------------- GLB 复用


def collect_reusable_pool(proj: dict, pdir: str) -> list:
    pool = []
    for r in proj.get("rooms", []):
        for it in (r.get("scene_spec") or {}).get("furniture", []):
            grel = it.get("glb_rel")
            if grel and os.path.isfile(os.path.join(pdir, grel)):
                pool.append({
                    "room_name": r.get("name", ""), "type": it.get("type", "other"),
                    "label": it.get("label", ""), "id": it.get("id"),
                    "glb_rel": grel, "glb_url": it.get("glb_url"),
                    "clean_rel": it.get("clean_rel"), "clean_url": it.get("clean_url"),
                    "crop_rel": it.get("crop_rel"), "crop_url": it.get("crop_url"),
                    "lux_task_id": it.get("lux_task_id"), "lux_version": it.get("lux_version", "G1-Turbo"),
                    "used": False,
                })
    return pool


def score_reusable_pair(e: dict, room_name: str, ftype: str, label: str, strict_same_room=False) -> int:
    COMPAT = (
        {"bookshelf", "cabinet"},
        {"coffee_table", "side_table"},
    )
    KWS = (
        "双人床", "单人床", "沙发", "餐桌", "餐椅", "电脑椅", "办公椅", "椅",
        "冰箱", "衣柜", "阳台柜", "储物柜", "转角", "书桌", "电视", "置物",
        "边桌", "圆桌", "茶几", "马桶", "热水器", "花洒", "橱柜", "地柜",
        "吊柜", "洗衣机", "洗手", "衣帽架", "晾衣架", "空调", "吊灯", "穿衣镜",
    )
    same = e["room_name"] == room_name
    if strict_same_room and not same:
        return -1
    etype = e["type"]
    elabel = e["label"]
    same_type = (etype == ftype)
    compat_type = any(etype in g and ftype in g for g in COMPAT)
    kw_hits = sum(1 for kw in KWS if kw in label and kw in elabel)
    if (ftype == "other" or etype == "other") and kw_hits == 0 and pc.norm_label(label) != pc.norm_label(elabel):
        return -1
    if ftype == "wall_cabinet" and "吊柜" in label and not any(k in elabel for k in ("吊柜", "橱柜", "柜")):
        return -1
    if not same_type and not compat_type and kw_hits == 0:
        return -1
    score = (12 if same_type else (7 if compat_type else 0)) + kw_hits * 8
    score += 15 if same else (5 if (e["room_name"] in room_name or room_name in e["room_name"]) else 0)
    min_req = 22 if strict_same_room else (12 if (same_type or kw_hits > 0) else 20)
    return score if score >= min_req else -1


def match_reusable(pool: list, room_name: str, ftype: str, label: str, strict_same_room=False):
    best, best_score = None, -1
    for e in pool:
        if e["used"]:
            continue
        score = score_reusable_pair(e, room_name, ftype, label, strict_same_room=strict_same_room)
        if score > best_score:
            best, best_score = e, score
    if best:
        best["used"] = True
    return best


# --------------------------------------------------------------------------- 编排


def run_build(pid: str, body: dict) -> dict:
    proj = load_project(pid)
    if not proj:
        raise ValueError("项目不存在")
    reparse = bool(body.get("reparseFloorplan"))
    force = bool(body.get("forceRebuild"))
    enqueue = bool(body.get("enqueueLux", True))
    only_room = body.get("roomId")
    skip_audit = bool(body.get("skipAudit"))
    pdir = project_dir(pid)

    fp_info = proj.get("floorplan") or {}
    fp_abs = None
    if fp_info.get("url"):
        cand = os.path.join(pdir, fp_info["url"].split("/")[-1])
        if os.path.isfile(cand):
            fp_abs = cand

    if reparse and fp_abs:
        summary, rooms, trace = pf.run_floorplan_pipeline(fp_abs, pdir, proj.get("rooms", []), proj,
                                                          do_audit=not skip_audit)
        proj["house_summary"] = summary
        proj["rooms"] = rooms
        proj.setdefault("pipeline", {})["floorplan"] = trace
        proj["built_3d"] = False
        with PROJECT_LOCK:
            save_project(pid, proj)

    total_photos = sum(len(r.get("photos", [])) for r in proj.get("rooms", []))
    if total_photos == 0:
        raise ValueError("请先至少为一个房间上传真实照片")

    reuse_glb = bool(body.get("reuseGlb", True))
    pool = collect_reusable_pool(proj, pdir) if ((force or reparse) and reuse_glb) else []
    todo = []
    for room in proj["rooms"]:
        if only_room and room["id"] != only_room:
            continue
        if not room.get("photos"):
            continue
        if not force and room.get("scene_spec") and room["scene_spec"].get("furniture"):
            continue
        todo.append(room)

    if todo:
        from concurrent.futures import ThreadPoolExecutor
        errors = []

        def _one(rm):
            try:
                rm["scene_spec"] = pr.build_room_spec(proj, rm, pdir, do_audit=not skip_audit)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{rm['name']}: {e}")
                rm["scene_spec"] = {"wall_hex": "#F2EEE7", "floor_hex": "#C9A176",
                                    "align_summary": f"解析失败：{e}", "furniture": []}

        with ThreadPoolExecutor(max_workers=min(2, len(todo))) as ex:
            list(ex.map(_one, todo))
        if errors:
            pc.pipeline_warning(proj, "build3d", "；".join(errors)[:600])

        if pool:
            id_remap = {}
            matched_items = set()
            for strict in (True, False):
                cands = []
                for rm in todo:
                    for it in (rm.get("scene_spec") or {}).get("furniture", []):
                        it_key = (rm["id"], it["id"])
                        if it.get("glb_rel") or it_key in matched_items:
                            continue
                        for e in pool:
                            if e["used"]:
                                continue
                            sc = score_reusable_pair(e, rm["name"], it["type"], it["label"], strict_same_room=strict)
                            if sc >= 0:
                                cands.append((sc, rm, it, e))
                cands.sort(key=lambda x: x[0], reverse=True)
                for sc, rm, it, e in cands:
                    it_key = (rm["id"], it["id"])
                    if e["used"] or it_key in matched_items:
                        continue
                    e["used"] = True
                    matched_items.add(it_key)
                    crop_cand = e["crop_rel"] or it.get("crop_rel")
                    pose = aholo.calibrate_glb_pose(
                        os.path.join(pdir, e["glb_rel"]),
                        it["type"],
                        it["size_m"],
                        crop_path=os.path.join(pdir, crop_cand) if crop_cand else None,
                        label=it.get("label", ""),
                        debug_dir=os.path.join(pdir, "debug", "glb_4view"),
                    )
                    old_id = it.get("id")
                    if old_id and e.get("id") and old_id != e["id"]:
                        id_remap[old_id] = e["id"]
                    it.update({
                        "id": e["id"], "crop_rel": e["crop_rel"] or it.get("crop_rel"),
                        "crop_url": e["crop_url"] or it.get("crop_url"),
                        "clean_rel": e.get("clean_rel"), "clean_url": e.get("clean_url"),
                        "glb_rel": e["glb_rel"], "glb_url": e["glb_url"],
                        "glb_fix_deg": pose["glb_fix_deg"], "glb_aligned_size": pose["glb_aligned_size"],
                        "lux_status": "succeeded", "lux_task_id": e.get("lux_task_id"),
                        "lux_version": e.get("lux_version", "G1-Turbo"),
                    })
            if id_remap:
                for rm in todo:
                    for it in (rm.get("scene_spec") or {}).get("furniture", []):
                        pid_ref = it.get("parent_id")
                        if pid_ref in id_remap:
                            it["parent_id"] = id_remap[pid_ref]

    proj["built_3d"] = True
    with PROJECT_LOCK:
        save_project(pid, proj)
    log_stage(proj, "build3d.done", "ok",
              f"{len(todo)} 间房重新解析，家具 {sum(len((r.get('scene_spec') or {}).get('furniture', [])) for r in proj['rooms'])} 件")
    with PROJECT_LOCK:
        save_project(pid, proj)
    if enqueue:
        enqueue_jobs(pid, custom_key=str(body.get("aholoApiKey") or ""))
    return load_project(pid)


# --------------------------------------------------------------------------- HTTP


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, code: int, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json_body(self, max_bytes=60 * 1024 * 1024):
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > max_bytes:
            raise ValueError("invalid payload size")
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _path(self):
        path = urlparse(self.path).path
        return path

    # ---------------------------------------------------------------- GET

    def do_GET(self):
        path = self._path()
        try:
            if path == "/api/pocketnest-v2/health":
                cfg = load_env_config()
                return self.send_json(200, {
                    "ok": True, "version": PIPELINE_VERSION,
                    "lux3d_ready": bool(cfg.get("AHOLO_API_KEY")),
                    "clean_plate": True, "img_to_3d": True,
                    "workers": WORKERS, "queue": LUX3D_QUEUE.qsize(),
                })

            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)$", path)
            if m:
                proj = load_project(m.group(1))
                return self.send_json(200, proj) if proj else self.send_json(404, {"error": "项目不存在"})

            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/pipeline$", path)
            if m:
                proj = load_project(m.group(1))
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                pipe = proj.get("pipeline") or {}
                items = []
                for r in proj.get("rooms", []):
                    for it in (r.get("scene_spec") or {}).get("furniture", []):
                        items.append({
                            "room": r["name"], "id": it["id"], "label": it.get("label"),
                            "clean": bool(it.get("clean_url")), "glb": bool(it.get("glb_url")),
                            "status": it.get("lux_status"), "error": it.get("lux_error"),
                            "views": it.get("views"), "confidence": it.get("confidence"),
                        })
                return self.send_json(200, {"version": pipe.get("version", PIPELINE_VERSION),
                                            "stage": pipe.get("stage"), "events": pipe.get("events", []),
                                            "warnings": pipe.get("warnings", []), "items": items,
                                            "queue": LUX3D_QUEUE.qsize()})

            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/pose/report$", path)
            if m:
                proj = load_project(m.group(1))
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                return self.send_json(200, {"pose_audit": (proj.get("pipeline") or {}).get("pose_audit")})

            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/floorplan/debug$", path)
            if m:
                proj = load_project(m.group(1))
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                fp = (proj.get("pipeline") or {}).get("floorplan") or {}
                stem = os.path.splitext(os.path.basename(((proj.get("floorplan") or {}).get("url") or "fp").split("/")[-1]))[0]
                pdir = project_dir(m.group(1))
                return self.send_json(200, {
                    "floorplan": proj.get("floorplan"),
                    "pass1": fp.get("pass1"), "pass2": fp.get("pass2"),
                    "scale": fp.get("scale"), "warnings": fp.get("warnings"),
                    "rooms": proj.get("rooms"),
                    "overlay_pass1": f"/api/pocketnest-v2/files/{m.group(1)}/debug/{stem}_overlay_pass1.jpg",
                    "overlay_final": f"/api/pocketnest-v2/files/{m.group(1)}/debug/{stem}_overlay_final.jpg",
                })

            m = re.match(r"^/api/pocketnest-v2/files/([a-zA-Z0-9_-]+)/(.+)$", path)
            if m:
                pid, rel = m.group(1), m.group(2)
                pdir = os.path.realpath(project_dir(pid))
                fpath = os.path.realpath(os.path.join(pdir, rel))
                if not fpath.startswith(pdir + os.sep) or not os.path.isfile(fpath):
                    return self.send_json(404, {"error": "文件不存在"})
                mime = mimetypes.guess_type(fpath)[0] or ("model/gltf-binary" if fpath.endswith(".glb") else "application/octet-stream")
                with open(fpath, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Cache-Control", "public, max-age=86400")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return

            return self.send_json(404, {"error": "未知接口"})
        except Exception as e:  # noqa: BLE001
            return self.send_json(500, {"error": str(e)})

    # ---------------------------------------------------------------- POST

    def do_POST(self):
        path = self._path()
        try:
            # 1) 新建项目
            if path == "/api/pocketnest-v2/projects":
                pid = "nest_" + uuid.uuid4().hex[:12]
                proj = {"id": pid, "created_at": int(time.time()), "house_summary": "",
                        "floorplan": None, "rooms": [], "built_3d": False,
                        "pipeline": {"version": PIPELINE_VERSION, "events": [], "warnings": []}}
                with PROJECT_LOCK:
                    save_project(pid, proj)
                return self.send_json(200, proj)

            # 2) 户型图上传 + 解析
            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/(floorplan|floorplan/reparse)$", path)
            if m:
                pid, action = m.group(1), m.group(2)
                proj = load_project(pid)
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                pdir = project_dir(pid)
                if action == "floorplan":
                    body = self.read_json_body()
                    data_url = body.get("dataUrl", "")
                    if not data_url.startswith("data:image/"):
                        return self.send_json(400, {"error": "请上传有效的图片文件"})
                    header, b64 = data_url.split(",", 1)
                    ext = ".png" if "png" in header else (".webp" if "webp" in header else ".jpg")
                    fp_rel = f"floorplan_{int(time.time())}{ext}"
                    fp_abs = os.path.join(pdir, fp_rel)
                    os.makedirs(pdir, exist_ok=True)
                    with open(fp_abs, "wb") as f:
                        f.write(base64.b64decode(b64))
                    proj["floorplan"] = {"name": body.get("name", "floorplan.png"),
                                         "url": f"/api/pocketnest-v2/files/{pid}/{fp_rel}"}
                else:
                    fp_abs = None
                    if (proj.get("floorplan") or {}).get("url"):
                        cand = os.path.join(pdir, proj["floorplan"]["url"].split("/")[-1])
                        if os.path.isfile(cand):
                            fp_abs = cand
                    if not fp_abs:
                        return self.send_json(400, {"error": "还没有上传户型图"})
                    body = {}

                summary, rooms, trace = pf.run_floorplan_pipeline(
                    fp_abs, pdir, proj.get("rooms", []), proj, do_audit=not bool(body.get("skipAudit")))
                proj["house_summary"] = summary
                proj["rooms"] = rooms
                proj.setdefault("pipeline", {})["floorplan"] = trace
                proj["built_3d"] = False
                with PROJECT_LOCK:
                    save_project(pid, proj)
                return self.send_json(200, load_project(pid))

            # 3) 手动增删房间
            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/rooms$", path)
            if m:
                pid = m.group(1)
                proj = load_project(pid)
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                body = self.read_json_body()
                if body.get("action") == "add":
                    idx = len(proj["rooms"])
                    proj["rooms"].append({
                        "id": f"rm_{uuid.uuid4().hex[:8]}",
                        "name": str(body.get("name") or "新房间").strip()[:18],
                        "category": body.get("category") if body.get("category") in pf.CATEGORY_ICON else "other",
                        "icon": "🏠", "rel_x": float((idx % 3 - 1) * 3.8), "rel_z": float((idx // 3) * 3.8),
                        "width_m": 3.8, "depth_m": 3.8, "window_dir": "S",
                        "openings": [{"kind": "door", "wall": "south", "offset": 0.75, "width_m": 0.9, "render_leaf": True}],
                        "photos": [], "scene_spec": None,
                    })
                elif body.get("action") == "delete":
                    proj["rooms"] = [r for r in proj["rooms"] if r["id"] != body.get("roomId")]
                with PROJECT_LOCK:
                    save_project(pid, proj)
                return self.send_json(200, proj)

            # 4) 上传照片
            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/rooms/([a-zA-Z0-9_-]+)/photos$", path)
            if m:
                pid, rid = m.group(1), m.group(2)
                proj = load_project(pid)
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                room = next((r for r in proj["rooms"] if r["id"] == rid), None)
                if not room:
                    return self.send_json(404, {"error": "房间不存在"})
                body = self.read_json_body()
                rdir = os.path.join(project_dir(pid), "rooms", rid)
                os.makedirs(rdir, exist_ok=True)
                for it in (body.get("photos") or [])[:12]:
                    durl = it.get("dataUrl", "")
                    if not durl.startswith("data:image/"):
                        continue
                    hdr, b64 = durl.split(",", 1)
                    ext = ".png" if "png" in hdr else (".webp" if "webp" in hdr else ".jpg")
                    phid = "ph_" + uuid.uuid4().hex[:10]
                    fname = phid + ext
                    with open(os.path.join(rdir, fname), "wb") as f:
                        f.write(base64.b64decode(b64))
                    room["photos"].append({
                        "id": phid, "name": str(it.get("name") or fname)[:40],
                        "rel_path": f"rooms/{rid}/{fname}",
                        "url": f"/api/pocketnest-v2/files/{pid}/rooms/{rid}/{fname}",
                        "uploaded_at": int(time.time()),
                    })
                room["scene_spec"] = None
                with PROJECT_LOCK:
                    save_project(pid, proj)
                return self.send_json(200, proj)

            # 5) 单房间解析
            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/rooms/([a-zA-Z0-9_-]+)/analyze$", path)
            if m:
                pid, rid = m.group(1), m.group(2)
                proj = load_project(pid)
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                room = next((r for r in proj["rooms"] if r["id"] == rid), None)
                if not room:
                    return self.send_json(404, {"error": "房间不存在"})
                body = {}
                try:
                    body = self.read_json_body()
                except Exception:
                    pass
                room["scene_spec"] = pr.build_room_spec(proj, room, project_dir(pid),
                                                        do_audit=bool(body.get("audit", True)))
                with PROJECT_LOCK:
                    save_project(pid, proj)
                if bool(body.get("enqueueLux", True)):
                    enqueue_jobs(pid)
                return self.send_json(200, load_project(pid))

            # 6) 全屋 3D 编排
            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/build3d$", path)
            if m:
                pid = m.group(1)
                body = {}
                try:
                    body = self.read_json_body()
                except Exception:
                    pass
                return self.send_json(200, run_build(pid, body))

            # 6.5) 多角度截图 + VLM 自纠偏闭环
            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/pose/audit$", path)
            if m:
                pid = m.group(1)
                body = {}
                try:
                    body = self.read_json_body()
                except Exception:
                    pass
                import pose_audit
                rep = pose_audit.run(pid, int(body.get("rounds", 2)))
                return self.send_json(200, rep)

            # 6.6) 回收卡住的 Lux3D 任务：Aholo 已完成但 worker 没写回的，补下载入库
            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/lux/recover$", path)
            if m:
                pid = m.group(1)
                proj = load_project(pid)
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                pdir = project_dir(pid)
                recovered = 0
                requeued = 0
                for r in proj.get("rooms", []):
                    for it in (r.get("scene_spec") or {}).get("furniture", []):
                        if it.get("glb_url") or it.get("lux_status") == "succeeded":
                            continue
                        tid = it.get("lux_task_id")
                        if tid:
                            try:
                                d = aholo.wait_task(int(tid), limit_s=8)
                                urls = aholo._output_urls(d)
                                glb = next((u for u in urls if u.split("?")[0].lower().endswith(".glb")), None)
                                if not glb:
                                    continue
                                raw_rel = f"models/{it['id']}.glb"
                                aholo._download(glb, os.path.join(pdir, raw_rel))
                                opt_rel = f"models_opt/{it['id']}.glb"
                                used = aholo.optimize_glb(os.path.join(pdir, raw_rel), os.path.join(pdir, opt_rel))
                                used_rel = opt_rel if used == os.path.join(pdir, opt_rel) else raw_rel
                                pose = aholo.calibrate_glb_pose(
                                    used,
                                    it.get("type", "other"),
                                    it.get("size_m"),
                                    crop_path=os.path.join(pdir, it["crop_rel"]) if it.get("crop_rel") else None,
                                    label=it.get("label", ""),
                                    debug_dir=os.path.join(pdir, "debug", "glb_4view"),
                                )
                                update_item(pid, r["id"], it["id"], lux_status="succeeded",
                                            glb_rel=used_rel,
                                            glb_url=f"/api/pocketnest-v2/files/{pid}/{used_rel}",
                                            glb_fix_deg=pose["glb_fix_deg"],
                                            glb_aligned_size=pose["glb_aligned_size"], lux_error=None)
                                recovered += 1
                            except Exception:
                                continue
                        elif it.get("lux_status") in ("queued", "generating", "uploading", "cleaning"):
                            update_item(pid, r["id"], it["id"], lux_status="pending")
                            requeued += 1
                if requeued:
                    enqueue_jobs(pid)
                return self.send_json(200, {"recovered": recovered, "requeued": requeued})

            # 7) 流水线单步调试
            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/pipeline/step$", path)
            if m:
                pid = m.group(1)
                body = self.read_json_body()
                step = str(body.get("step") or "")
                proj = load_project(pid)
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                pdir = project_dir(pid)
                result = {"step": step}
                if step in ("floorplan_parse", "floorplan_audit"):
                    fp_abs = os.path.join(pdir, ((proj.get("floorplan") or {}).get("url") or "").split("/")[-1])
                    if not os.path.isfile(fp_abs):
                        return self.send_json(400, {"error": "还没有上传户型图"})
                    draft = pf.parse_floorplan_draft(fp_abs)
                    result["pass1"] = draft
                    if step == "floorplan_audit":
                        rooms_raw = [c for c in (pf._clean_room(x, i) for i, x in enumerate(draft.get("rooms") or [])) if c]
                        stem = os.path.splitext(os.path.basename(fp_abs))[0]
                        overlay = os.path.join(pdir, f"debug/{stem}_overlay_pass1.jpg")
                        pc.draw_floorplan_overlay(fp_abs, rooms_raw, overlay)
                        result["pass2"] = pf.audit_floorplan(fp_abs, overlay, draft)
                        result["overlay_url"] = f"/api/pocketnest-v2/files/{pid}/debug/{stem}_overlay_pass1.jpg"
                elif step == "room_analyze":
                    rid = body.get("roomId")
                    room = next((r for r in proj["rooms"] if r["id"] == rid), None)
                    if not room:
                        return self.send_json(400, {"error": "roomId 不存在"})
                    spec = pr.build_room_spec(proj, room, pdir, do_audit=body.get("audit", True))
                    room["scene_spec"] = spec
                    result["scene_spec"] = spec
                elif step in ("clean_plate", "img_to_3d"):
                    rid, iid = body.get("roomId"), body.get("itemId")
                    room, item = find_item(proj, rid, iid)
                    if not item:
                        return self.send_json(400, {"error": "itemId 不存在"})
                    crop_abs = os.path.join(pdir, item.get("crop_rel") or "")
                    if not os.path.isfile(crop_abs):
                        return self.send_json(400, {"error": "裁图不存在"})
                    if step == "clean_plate":
                        out_rel = f"clean/{iid}.png"
                        clean_src = os.path.join(pdir, item.get("clean_source_rel") or item["crop_rel"])
                        if not os.path.isfile(clean_src):
                            clean_src = crop_abs
                        res = aholo.clean_plate(clean_src, item, os.path.join(pdir, out_rel))
                        result["clean"] = res
                        if res.get("ok"):
                            update_item(pid, rid, iid, clean_rel=out_rel,
                                        clean_url=f"/api/pocketnest-v2/files/{pid}/{out_rel}",
                                        clean_error=None)
                    else:
                        src = os.path.join(pdir, item["clean_rel"]) if item.get("clean_rel") else crop_abs
                        url = aholo.upload_image(src)
                        raw_rel = f"models/{iid}.glb"
                        res = aholo.image_to_3d(url, os.path.join(pdir, raw_rel),
                                                version="G1" if item.get("is_glass") else "G1-Turbo")
                        opt_rel = f"models_opt/{iid}.glb"
                        used_abs = aholo.optimize_glb(os.path.join(pdir, raw_rel), os.path.join(pdir, opt_rel))
                        glb_rel = opt_rel if used_abs == os.path.join(pdir, opt_rel) else raw_rel
                        pose = aholo.calibrate_glb_pose(
                            os.path.join(pdir, glb_rel),
                            item.get("type", "other"),
                            item.get("size_m"),
                            crop_path=crop_abs,
                            label=item.get("label", ""),
                            debug_dir=os.path.join(pdir, "debug", "glb_4view"),
                        )
                        update_item(pid, rid, iid, lux_status="succeeded", glb_rel=glb_rel,
                                    glb_url=f"/api/pocketnest-v2/files/{pid}/{glb_rel}",
                                    lux_task_id=res.get("task_id"), glb_fix_deg=pose["glb_fix_deg"],
                                    glb_aligned_size=pose["glb_aligned_size"], lux_error=None)
                        result["glb"] = res
                else:
                    return self.send_json(400, {"error": f"未知 step: {step}"})
                # clean_plate / img_to_3d 已经用 update_item 落盘，这里只落 room_analyze 的内存修改
                if step == "room_analyze":
                    with PROJECT_LOCK:
                        save_project(pid, proj)
                return self.send_json(200, result)

            # 8) 单件家具操作
            m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/rooms/([a-zA-Z0-9_-]+)/furniture$", path)
            if m:
                pid, rid = m.group(1), m.group(2)
                proj = load_project(pid)
                if not proj:
                    return self.send_json(404, {"error": "项目不存在"})
                room = next((r for r in proj["rooms"] if r["id"] == rid), None)
                if not room:
                    return self.send_json(404, {"error": "房间不存在"})
                if not room.get("scene_spec"):
                    room["scene_spec"] = {"wall_hex": "#F2EEE7", "floor_hex": "#C9A176", "furniture": []}
                body = self.read_json_body()
                act = body.get("action")
                rw, rd = float(room.get("width_m", 3.4)), float(room.get("depth_m", 3.4))
                if act in ("rotate", "rotate_step"):
                    iid, step = body.get("itemId"), int(body.get("stepDeg", 90))
                    for it in room["scene_spec"].get("furniture", []):
                        if it["id"] != iid:
                            continue
                        rot = (int(it.get("rot_deg", 0)) + step) % 360
                        it["rot_deg"] = rot
                        import math
                        fw, fh, fd = it.get("size_m", [0.8, 0.8, 0.6])
                        rad = math.radians(rot)
                        wx = abs(fw * math.cos(rad)) + abs(fd * math.sin(rad))
                        wz = abs(fw * math.sin(rad)) + abs(fd * math.cos(rad))
                        it["world_box_m"] = [round(wx, 2), round(fh, 2), round(wz, 2)]
                        min_x, max_x = -rw / 2 + wx / 2 + 0.04, rw / 2 - wx / 2 - 0.04
                        min_z, max_z = -rd / 2 + wz / 2 + 0.04, rd / 2 - wz / 2 - 0.04
                        xm = float(it.get("xm", 0))
                        zm = float(it.get("zm", 0))
                        it["xm"] = round(max(min_x, min(max_x, xm)) if min_x <= max_x else 0.0, 3)
                        it["zm"] = round(max(min_z, min(max_z, zm)) if min_z <= max_z else 0.0, 3)
                        it["pos_x"], it["pos_z"] = round(it["xm"] / rw, 4), round(it["zm"] / rd, 4)
                elif act == "move":
                    iid = body.get("itemId")
                    for it in room["scene_spec"].get("furniture", []):
                        if it["id"] != iid:
                            continue
                        wx, _, wz = it.get("world_box_m") or it.get("size_m") or [0.6, 0.8, 0.6]
                        min_x, max_x = -rw / 2 + wx / 2 + 0.03, rw / 2 - wx / 2 - 0.03
                        min_z, max_z = -rd / 2 + wz / 2 + 0.03, rd / 2 - wz / 2 - 0.03
                        it["xm"] = round(max(min_x, min(max_x, float(body.get("xm", 0)))) if min_x <= max_x else 0.0, 3)
                        it["zm"] = round(max(min_z, min(max_z, float(body.get("zm", 0)))) if min_z <= max_z else 0.0, 3)
                        it["pos_x"], it["pos_z"] = round(it["xm"] / rw, 4), round(it["zm"] / rd, 4)
                elif act == "retry_lux3d":
                    for it in room["scene_spec"].get("furniture", []):
                        if it["id"] == body.get("itemId"):
                            it.update({"lux_status": "pending", "glb_url": None, "glb_rel": None,
                                       "lux_error": None, "clean_error": None})
                elif act == "reset_lux":
                    # 调试/对比用：清掉没有走新 clean-plate 链路的旧 GLB，重新排队
                    only_legacy = bool(body.get("onlyLegacy", True))
                    n = 0
                    for r in proj.get("rooms", []):
                        for it in (r.get("scene_spec") or {}).get("furniture", []):
                            if only_legacy and it.get("lux_clean_used"):
                                continue
                            if not it.get("glb_url") and not it.get("clean_url"):
                                continue
                            it.update({"lux_status": "pending", "glb_url": None, "glb_rel": None,
                                       "clean_url": None, "clean_rel": None,
                                       "lux_error": None, "clean_error": None,
                                       "lux_clean_used": False})
                            n += 1
                    with PROJECT_LOCK:
                        save_project(pid, proj)
                    enqueue_jobs(pid)
                    return self.send_json(200, {"reset": n, "project": load_project(pid)})
                elif act == "add_single":
                    durl = body.get("dataUrl", "")
                    if not durl.startswith("data:image/"):
                        return self.send_json(400, {"error": "请上传单品照片"})
                    _, b64 = durl.split(",", 1)
                    iid = f"item_{uuid.uuid4().hex[:8]}"
                    crop_rel = f"crops/{iid}.jpg"
                    crop_abs = os.path.join(project_dir(pid), crop_rel)
                    os.makedirs(os.path.dirname(crop_abs), exist_ok=True)
                    with open(crop_abs, "wb") as f:
                        f.write(base64.b64decode(b64))
                    label = str(body.get("label") or "新 3D 物件").strip()[:20]
                    room["scene_spec"].setdefault("furniture", []).append({
                        "id": iid, "type": "other", "label": label, "color_hex": "#D8A36C",
                        "appearance": label, "mount": "floor", "wall_anchor": "center",
                        "xm": 0.0, "zm": 0.0, "elevation_m": 0.0, "pos_x": 0.0, "pos_z": 0.0,
                        "size_m": [0.70, 0.75, 0.60], "world_box_m": [0.70, 0.75, 0.60],
                        "rot_deg": 0, "glb_fix_deg": 0.0, "is_glass": bool(body.get("isGlass")),
                        "lux3d_prompt": label, "crop_rel": crop_rel,
                        "crop_url": f"/api/pocketnest-v2/files/{pid}/{crop_rel}",
                        "clean_source_rel": crop_rel,
                        "clean_source_url": f"/api/pocketnest-v2/files/{pid}/{crop_rel}",
                        "clean_rel": None, "clean_url": None, "clean_error": None,
                        "lux_status": "pending", "glb_url": None,
                    })
                with PROJECT_LOCK:
                    save_project(pid, proj)
                if act in ("retry_lux3d", "add_single"):
                    enqueue_jobs(pid, custom_key=str(body.get("aholoApiKey") or ""))
                return self.send_json(200, load_project(pid))

            return self.send_json(404, {"error": "未知接口"})
        except Exception as e:  # noqa: BLE001
            return self.send_json(500, {"error": str(e)})

    # ---------------------------------------------------------------- DELETE

    def do_DELETE(self):
        path = self._path()
        m = re.match(r"^/api/pocketnest-v2/projects/([a-zA-Z0-9_-]+)/rooms/([a-zA-Z0-9_-]+)/photos/([a-zA-Z0-9_-]+)$", path)
        if not m:
            return self.send_json(404, {"error": "未知删除接口"})
        pid, rid, phid = m.group(1), m.group(2), m.group(3)
        proj = load_project(pid)
        if not proj:
            return self.send_json(404, {"error": "项目不存在"})
        room = next((r for r in proj["rooms"] if r["id"] == rid), None)
        if not room:
            return self.send_json(404, {"error": "房间不存在"})
        target = next((p for p in room["photos"] if p["id"] == phid), None)
        if target and target.get("rel_path"):
            fpath = os.path.join(project_dir(pid), target["rel_path"])
            if os.path.isfile(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
        room["photos"] = [p for p in room["photos"] if p["id"] != phid]
        room["scene_spec"] = None
        with PROJECT_LOCK:
            save_project(pid, proj)
        return self.send_json(200, proj)


if __name__ == "__main__":
    for i in range(max(1, WORKERS)):
        threading.Thread(target=lux_worker_loop, args=(i,), daemon=True).start()
    host = os.environ.get("PN_HOST", "127.0.0.1")
    port = int(os.environ.get("PN_PORT", "18792"))
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"[PocketNest-{PIPELINE_VERSION}] listening on http://{host}:{port} workers={WORKERS}", flush=True)
    srv.serve_forever()
