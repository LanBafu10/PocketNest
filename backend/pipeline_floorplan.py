#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""户型图解析流水线（PocketNest v3 阶段 1）。

  pass1  parse_floorplan_draft()    中性提示词，只提取图上真实存在的空间
  audit  audit_floorplan()          把识别框画回户型图，让模型逐项纠错
  scale  calibrate_scale()          房间尺寸尺度 × 全屋面积尺度，交叉校验
  snap   snap_and_validate()        门洞合缝、共享墙吸附、重叠裁剪、门窗连通性校验

与旧版的差异：
- 旧提示词把小家的房间名/阳台/入户门位置写死，模型会迎合提示词造出「连通阳台」；
  这里改成中性提示词，并且二审只做「保守修正」——与一审 IoU 不足 0.65 的坐标一律不采纳。
- 旧版只用一个 ×1.32 的面积比例尺；这里用「每间房估算尺寸」和「全屋面积」两套独立尺度，
  先剔除长宽比自相矛盾的房间，再取几何平均。
- 旧版聚类后没有真正处理包围盒侵入；这里对「小房间被大房间包住」的情况裁大房间，
  对普通贴边重叠裁小房间，并在有门相连的两面墙之间合缝。
"""

from __future__ import annotations

import math
import os
import uuid

import floorplan_refine as refine
from pipeline_common import (
    FP_VLM_MODEL,
    call_vlm_json,
    clamp,
    draw_floorplan_overlay,
    image_size,
    log_stage,
    median,
    norm_label,
    pipeline_warning,
    rect_overlap,
)

CATEGORY_ICON = {
    "bedroom": "🛏️", "living": "🛋️", "dining": "🍽️", "kitchen": "🍳", "bath": "🛁",
    "study": "📚", "balcony": "🌿", "entry": "🚪", "storage": "📦", "other": "🏠",
}

WALLS = ("north", "south", "west", "east")
OP_KINDS = ("door", "sliding_door", "open_arch", "open_passage", "entry_door", "window")


FP_PROMPT_PASS1 = """你是一位严谨的建筑户型图解析器。请只依据这张 2D 户型平面图上真实可见的墙线、门扇圆弧、窗线和文字标注提取空间，不要臆测图上不存在的房间。

【坐标系】图片左上角为原点，x 向右，y 向下。bbox_1000 = [x1, y1, x2, y2]，取值 0~1000，且 x1 < x2、y1 < y2。框必须是该空间「内墙围合」的净区域，不要把墙体厚度、隔壁房间或公共走道框进来。

【任务】
1. 逐个列出图中每一个封闭或半封闭空间：卧室、客厅、餐厅、厨房、卫生间、阳台、过道/干湿分离洗手台区、储物间、玄关等。图上没有文字标注但明显围合（或夹在客厅/卫生间/厨房之间的过渡过道/干区）也要列出。
2. 每个空间给出：
   - name：优先用图上文字；没有文字就用功能名（如「过道」「干区」「阳台」）
   - category：bedroom | living | dining | kitchen | bath | study | balcony | entry | storage | other
   - bbox_1000：见上
   - est_width_m：东西向净宽（米）；est_depth_m：南北向净深（米）。用室内门宽（约 0.8~0.9m）、图上标注或常见家具尺度推算，保留一位小数
   - window_dir：主要采光面朝向 N | S | E | W | NONE
   - openings：该空间四壁上的门/窗/垭口，每项 {{kind, wall, offset, width_m, target_room}}
       * kind：door 平开门 | sliding_door 推拉门 | open_arch 无门垭口 | open_passage 敞开过道 | entry_door 入户门 | window 窗
       * wall：north | south | west | east
       * offset：0~1，南北墙 0=最西端 1=最东端；东西墙 0=最北端 1=最南端
       * width_m：门/窗净宽（米）
       * target_room：该门通向的房间名（窗填「户外」）
3. **严禁矩形框互相重叠**：
   - 相邻房间的共享墙坐标必须对齐、不能吞进隔壁房间。
   - 如果客厅/餐厅是 L 形（含入户玄关、干湿分离洗手过道、走廊），请拆成「客厅」+「过道/干区」两个紧邻但不重叠的矩形分别框选，保证每个门洞都能准确落在两个连通房间的共享墙上。
4. 只输出图上真实存在的空间。拿不准的空间宁可不输出，也不要为了凑数编造。

另外给出：
- north_direction：指北针相对图片的方向 up | down | left | right | NONE
- estimated_area_m2：整套住宅的套内使用面积估算（不是建筑面积）
- house_summary：一句话概括户型

严格输出 JSON（不要多余文字）：
{{
  "house_summary": "两室一厅一厨一卫",
  "estimated_area_m2": 62.0,
  "north_direction": "up",
  "rooms": [
    {{
      "name": "卧室A",
      "category": "bedroom",
      "bbox_1000": [250, 60, 390, 230],
      "est_width_m": 2.7,
      "est_depth_m": 3.2,
      "window_dir": "N",
      "openings": [
        {{"kind": "door", "wall": "south", "offset": 0.70, "width_m": 0.9, "target_room": "客厅"}},
        {{"kind": "window", "wall": "north", "offset": 0.50, "width_m": 1.4, "target_room": "户外"}}
      ]
    }}
  ]
}}"""

FP_PROMPT_AUDIT = """第 1 张图是原始户型图，第 2 张图是上一轮识别结果叠加在户型图上的审核图（彩色框 + 编号.房间名）。

请像审图员一样逐项核对，并输出修正后的完整结果：
1. 【框是否贴合】每个框是否恰好覆盖该空间的净区域？吞进隔壁房间、走道、墙体，或只框住一部分的，请修正 bbox_1000。
2. 【有没有漏】图上明显存在、但上一轮没框出来的空间（过道/干区、阳台、储物间、卫生间、玄关）请补上。
3. 【有没有多】上一轮是否凭空造出了图上不存在的房间？请删除。
4. 【共享墙】相邻房间的共享墙坐标应一致，房间之间不应大面积重叠。
5. 【门窗】门/窗是否在正确的墙上？offset 与 width_m 是否合理（室内门 0.7~1.0m，入户门 0.9~1.1m，窗 0.6~3.0m）？
6. 【尺寸】est_width_m / est_depth_m 是否与 bbox 的长宽比一致？矛盾时以图上比例为准重估。

输出格式与上一轮一致（house_summary / estimated_area_m2 / north_direction / rooms），并额外给出：
- audit_notes：字符串数组，每条写「改了哪个房间、改了什么、为什么」。

不要解释，只输出 JSON。"""


# --------------------------------------------------------------------------- 清洗


def _norm_bbox(bb) -> list | None:
    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bb]
    except Exception:
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1, y1 = clamp(x1, 0, 1000), clamp(y1, 0, 1000)
    x2, y2 = clamp(x2, 0, 1000), clamp(y2, 0, 1000)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def _clean_room(raw: dict, idx: int) -> dict | None:
    name = str(raw.get("name") or f"房间{idx + 1}").strip()[:18]
    cat = str(raw.get("category") or "other").lower().strip()
    if cat not in CATEGORY_ICON:
        cat = "other"
    bb = _norm_bbox(raw.get("bbox_1000"))
    if not bb:
        return None
    ops = []
    for op in (raw.get("openings") or [])[:10]:
        if not isinstance(op, dict):
            continue
        kind = str(op.get("kind") or "door").lower().strip()
        if kind not in OP_KINDS:
            kind = "door"
        wall = str(op.get("wall") or "").lower().strip()
        if wall not in WALLS:
            continue
        try:
            off = float(op.get("offset", 0.5))
            ow = float(op.get("width_m") or 0.9)
        except Exception:
            off, ow = 0.5, 0.9
        ops.append({
            "kind": kind, "wall": wall,
            "offset": round(clamp(off, 0.05, 0.95), 3),
            "width_m": round(clamp(ow, 0.4, 4.0), 2),
            "target": str(op.get("target_room") or "").strip()[:18],
            "render_leaf": True,
        })
    try:
        ew, ed = float(raw.get("est_width_m") or 0), float(raw.get("est_depth_m") or 0)
    except Exception:
        ew, ed = 0.0, 0.0
    return {
        "name": name, "category": cat, "bbox_1000": bb,
        "est_width_m": ew if 0.4 <= ew <= 30 else 0.0,
        "est_depth_m": ed if 0.4 <= ed <= 30 else 0.0,
        "window_dir": str(raw.get("window_dir") or "NONE").upper()[:4],
        "openings": ops,
    }


def _rotate_bbox(bb: list, north: str) -> list:
    x1, y1, x2, y2 = bb
    if north == "down":
        return [1000 - x2, 1000 - y2, 1000 - x1, 1000 - y1]
    if north == "right":
        return [y1, 1000 - x2, y2, 1000 - x1]
    if north == "left":
        return [1000 - y2, x1, 1000 - y1, x2]
    return bb


def _rotate_wall(wall: str, north: str) -> str:
    if north == "down":
        return {"north": "south", "south": "north", "east": "west", "west": "east"}[wall]
    if north == "right":
        return {"east": "north", "north": "west", "west": "south", "south": "east"}[wall]
    if north == "left":
        return {"west": "north", "north": "east", "east": "south", "south": "west"}[wall]
    return wall


# --------------------------------------------------------------------------- pass1 / pass2


def parse_floorplan_draft(fp_abs: str, model: str = None) -> dict:
    return call_vlm_json(FP_PROMPT_PASS1, [fp_abs], timeout=180,
                         model=model or FP_VLM_MODEL, max_tokens=16384)


def audit_floorplan(fp_abs: str, overlay_abs: str, draft: dict, model: str = None) -> dict:
    import json
    note = ("上一轮识别结果如下（供对照，不要直接复制错误）：\n"
            + json.dumps(draft.get("rooms") or [], ensure_ascii=False, separators=(",", ":"))[:9000]
            + "\n\n请输出修正后的完整 JSON。")
    return call_vlm_json(FP_PROMPT_AUDIT + note, [fp_abs, overlay_abs], timeout=240,
                         model=model or FP_VLM_MODEL, max_tokens=16384)


# --------------------------------------------------------------------------- 尺度


def calibrate_scale(rooms: list, img_w: int, img_h: int, est_area: float):
    dims_a, per_room = [], []
    for r in rooms:
        x1, y1, x2, y2 = r["bbox_1000"]
        pw = (x2 - x1) / 1000.0 * img_w
        ph = (y2 - y1) / 1000.0 * img_h
        ew, ed = r.get("est_width_m", 0), r.get("est_depth_m", 0)
        if ew > 0.3 and ed > 0.3 and pw > 20 and ph > 20:
            ratio = (ew / ed) / max(0.01, pw / ph)
            ok = 0.60 <= ratio <= 1.67
            per_room.append({"name": r["name"], "est_aspect": round(ew / ed, 2),
                             "bbox_aspect": round(pw / ph, 2), "ratio": round(ratio, 2), "used": ok})
            if ok:
                dims_a += [ew / pw, ed / ph]
        elif ew > 0.3 and pw > 20:
            dims_a.append(ew / pw)
        elif ed > 0.3 and ph > 20:
            dims_a.append(ed / ph)
    scale_a = median(dims_a, 0.0)

    px_area = sum(max(1.0, (r["bbox_1000"][2] - r["bbox_1000"][0]) / 1000.0 * img_w)
                  * max(1.0, (r["bbox_1000"][3] - r["bbox_1000"][1]) / 1000.0 * img_h) for r in rooms)
    # 简版户型图内墙线条较粗，纯内净像素面积往往略小于真实套内开间；按 1.15 倍展示余量对齐总面积
    scale_b = math.sqrt(max(1.0, float(est_area or 0) * 1.15) / px_area) if (px_area > 0 and float(est_area or 0) > 0) else 0.0

    info = {
        "scale_from_room_dims": round(scale_a, 6), "scale_from_total_area": round(scale_b, 6),
        "est_area_m2": round(float(est_area or 0), 1), "sum_room_px_area": round(px_area, 1),
        "img_size": [img_w, img_h], "room_aspect_check": per_room, "choice": "",
    }
    if scale_a > 0 and scale_b > 0:
        ratio = scale_b / scale_a
        info["ratio_b_over_a"] = round(ratio, 3)
        if 0.68 <= ratio <= 1.48:
            base_scale, info["choice"] = math.sqrt(scale_a * scale_b), "geometric_mean"
        else:
            base_scale, info["choice"] = scale_a, "room_dims(area_mismatch)"
    elif scale_a > 0:
        base_scale, info["choice"] = scale_a, "room_dims"
    elif scale_b > 0:
        base_scale, info["choice"] = scale_b, "total_area"
    else:
        base_scale, info["choice"] = 0.012, "fallback_default"

    # 舒适人体工学开间校准：确保主卧/主起居室具备真实可居住宽度（主卧短边 ~3.2-4.2m），
    # 同时严格保持全屋等比缩放不扭曲长宽比。
    bed_px = []
    living_px = []
    for r in rooms:
        x1, y1, x2, y2 = r["bbox_1000"]
        pw = max(1.0, (x2 - x1) / 1000.0 * img_w)
        ph = max(1.0, (y2 - y1) / 1000.0 * img_h)
        short_px = min(pw, ph)
        area_px = pw * ph
        if r.get("category") == "bedroom":
            bed_px.append((area_px, short_px))
        elif r.get("category") == "living":
            living_px.append((area_px, short_px))

    comfort_scale = 1.0
    ref_short_px = 0.0
    if bed_px:
        bed_px.sort(reverse=True)
        ref_short_px = bed_px[0][1]
    elif living_px:
        living_px.sort(reverse=True)
        ref_short_px = living_px[0][1]

    if ref_short_px > 10:
        cur_short_m = ref_short_px * base_scale
        if cur_short_m < 3.45:
            needed = 3.52 / ref_short_px
            comfort_scale = clamp(needed / max(1e-6, base_scale), 1.10, 1.32)
            # 防止把主卧撑得过宽（上限 4.15m）
            if ref_short_px * base_scale * comfort_scale > 4.15:
                comfort_scale = max(1.0, 4.15 / (ref_short_px * base_scale))
        elif cur_short_m < 3.85:
            comfort_scale = 1.08

    scale = base_scale * comfort_scale
    info["base_meter_per_px"] = round(base_scale, 6)
    info["comfort_scale"] = round(comfort_scale, 3)
    info["meter_per_px"] = round(scale, 6)
    if ref_short_px > 10:
        info["main_room_short_m"] = round(ref_short_px * scale, 2)
    return scale, info


# --------------------------------------------------------------------------- 几何


def _build_room(item: dict, cid: str, cx: float, cz: float, scale: float, img_w: int, img_h: int) -> dict:
    x1, y1, x2, y2 = item["bbox_1000"]
    mx1, my1 = (x1 / 1000.0 * img_w - cx) * scale, (y1 / 1000.0 * img_h - cz) * scale
    mx2, my2 = (x2 / 1000.0 * img_w - cx) * scale, (y2 / 1000.0 * img_h - cz) * scale
    win = item.get("window_dir") or "NONE"
    if win not in ("N", "S", "E", "W", "NONE"):
        win = "NONE"
    return {
        "id": cid, "name": item["name"], "category": item["category"],
        "icon": CATEGORY_ICON.get(item["category"], "🏠"),
        "rel_x": round((mx1 + mx2) / 2, 2), "rel_z": round((my1 + my2) / 2, 2),
        "width_m": round(mx2 - mx1, 2), "depth_m": round(my2 - my1, 2),
        "window_dir": win, "openings": [dict(o) for o in item.get("openings") or []],
        "photos": [], "scene_spec": None, "bbox_1000": item["bbox_1000"],
        "est_width_m": item.get("est_width_m"), "est_depth_m": item.get("est_depth_m"),
        "_bounds": [mx1, my1, mx2, my2],
    }


def _refresh(room: dict):
    b = room["_bounds"]
    room["width_m"] = round(max(0.6, b[2] - b[0]), 2)
    room["depth_m"] = round(max(0.6, b[3] - b[1]), 2)
    room["rel_x"] = round((b[0] + b[2]) / 2, 2)
    room["rel_z"] = round((b[1] + b[3]) / 2, 2)


def _trim_to(room: dict, other: dict, min_dim: float = 0.60):
    """把 room 的侵入边退回到 other 的墙边，绝不把另一边推回重叠。"""
    b, o = room["_bounds"], other["_bounds"]
    ox, oz = rect_overlap(b, o)
    if ox <= 0.004 or oz <= 0.004:
        return
    cx, cz = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    ocx, ocz = (o[0] + o[2]) / 2, (o[1] + o[3]) / 2
    if ox <= oz:
        if ocx >= cx:
            b[2] = min(b[2], o[0])
            if b[2] - b[0] < min_dim:
                b[0] = min(b[0], b[2] - min_dim)
        else:
            b[0] = max(b[0], o[2])
            if b[2] - b[0] < min_dim:
                b[2] = max(b[2], b[0] + min_dim)
    else:
        if ocz >= cz:
            b[3] = min(b[3], o[1])
            if b[3] - b[1] < min_dim:
                b[1] = min(b[1], b[3] - min_dim)
        else:
            b[1] = max(b[1], o[3])
            if b[3] - b[1] < min_dim:
                b[3] = max(b[3], b[1] + min_dim)
    _refresh(room)


def _resolve_overlaps(rooms: list, eps: float = 0.004) -> list:
    """严格消除任意两间房的内部矩形重叠（保证所有成对交集面积为 0）。"""
    for _ in range(12):
        changed = False
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                a, b = rooms[i], rooms[j]
                ox, oz = rect_overlap(a["_bounds"], b["_bounds"])
                if ox <= eps or oz <= eps:
                    continue
                area_a = max(0.1, (a["_bounds"][2] - a["_bounds"][0]) * (a["_bounds"][3] - a["_bounds"][1]))
                area_b = max(0.1, (b["_bounds"][2] - b["_bounds"][0]) * (b["_bounds"][3] - b["_bounds"][1]))
                if (ox * oz) / max(0.1, min(area_a, area_b)) > 0.35:
                    _trim_to(a if area_a > area_b else b, b if area_a > area_b else a)
                else:
                    _trim_to(a if area_a <= area_b else b, b if area_a <= area_b else a)
                changed = True
        if not changed:
            break
    # 最后一轮硬裁剪（不向外推，彻底清零任何残余微重叠）
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b = rooms[i], rooms[j]
            ox, oz = rect_overlap(a["_bounds"], b["_bounds"])
            if ox <= eps or oz <= eps:
                continue
            area_a = max(0.1, (a["_bounds"][2] - a["_bounds"][0]) * (a["_bounds"][3] - a["_bounds"][1]))
            area_b = max(0.1, (b["_bounds"][2] - b["_bounds"][0]) * (b["_bounds"][3] - b["_bounds"][1]))
            victim, keep = (a, b) if area_a <= area_b else (b, a)
            vb, kb = victim["_bounds"], keep["_bounds"]
            if ox <= oz:
                if (kb[0] + kb[2]) / 2 >= (vb[0] + vb[2]) / 2:
                    vb[2] = min(vb[2], kb[0])
                else:
                    vb[0] = max(vb[0], kb[2])
            else:
                if (kb[1] + kb[3]) / 2 >= (vb[1] + vb[3]) / 2:
                    vb[3] = min(vb[3], kb[1])
                else:
                    vb[1] = max(vb[1], kb[3])
            _refresh(victim)
    return rooms


def _snap_door_walls(rooms: list, tol: float = 0.40) -> int:
    """门正好开向对面房间时，把两面墙的缝对半合上（只处理 <40cm 的缝）。"""
    done, moved = set(), 0
    for r in rooms:
        for op in r.get("openings") or []:
            if op.get("kind") not in ("door", "sliding_door", "open_arch", "open_passage", "entry_door"):
                continue
            wall, b = op.get("wall"), r["_bounds"]
            ow = float(op.get("width_m") or 0.9)
            best = None
            if wall in ("north", "south"):
                c = b[0] + float(op.get("offset", 0.5)) * (b[2] - b[0])
                for o in rooms:
                    if o["id"] == r["id"]:
                        continue
                    ob = o["_bounds"]
                    gap = (ob[1] - b[3]) if wall == "south" else (b[1] - ob[3])
                    if gap < -0.10 or gap > tol:
                        continue
                    if min(b[2], ob[2]) - max(b[0], ob[0]) < max(0.6, ow * 0.6):
                        continue
                    if not (max(b[0], ob[0]) <= c <= min(b[2], ob[2])):
                        continue
                    if best is None or abs(gap) < abs(best[0]):
                        best = (gap, o)
                if best and ("x", r["id"], best[1]["id"]) not in done:
                    o, ob = best[1], best[1]["_bounds"]
                    mid = ((b[3] if wall == "south" else b[1]) + (ob[1] if wall == "south" else ob[3])) / 2
                    if ((mid - b[1]) if wall == "south" else (b[3] - mid)) < 0.65:
                        continue
                    if ((ob[3] - mid) if wall == "south" else (mid - ob[1])) < 0.65:
                        continue
                    if wall == "south":
                        b[3], ob[1] = mid, mid
                    else:
                        b[1], ob[3] = mid, mid
                    done.update({("x", r["id"], o["id"]), ("x", o["id"], r["id"])})
                    moved += 1
            else:
                c = b[1] + float(op.get("offset", 0.5)) * (b[3] - b[1])
                for o in rooms:
                    if o["id"] == r["id"]:
                        continue
                    ob = o["_bounds"]
                    gap = (ob[0] - b[2]) if wall == "east" else (b[0] - ob[2])
                    if gap < -0.10 or gap > tol:
                        continue
                    if min(b[3], ob[3]) - max(b[1], ob[1]) < max(0.6, ow * 0.6):
                        continue
                    if not (max(b[1], ob[1]) <= c <= min(b[3], ob[3])):
                        continue
                    if best is None or abs(gap) < abs(best[0]):
                        best = (gap, o)
                if best and ("z", r["id"], best[1]["id"]) not in done:
                    o, ob = best[1], best[1]["_bounds"]
                    mid = ((b[2] if wall == "east" else b[0]) + (ob[0] if wall == "east" else ob[2])) / 2
                    if ((mid - b[0]) if wall == "east" else (b[2] - mid)) < 0.65:
                        continue
                    if ((ob[2] - mid) if wall == "east" else (mid - ob[0])) < 0.65:
                        continue
                    if wall == "east":
                        b[2], ob[0] = mid, mid
                    else:
                        b[0], ob[2] = mid, mid
                    done.update({("z", r["id"], o["id"]), ("z", o["id"], r["id"])})
                    moved += 1
    for r in rooms:
        _refresh(r)
    return moved


def _find_neighbor(room: dict, rooms: list, wall: str, op: dict):
    b = room["_bounds"]
    ow = float(op.get("width_m") or 0.9)
    span = (b[2] - b[0]) if wall in ("north", "south") else (b[3] - b[1])
    c = (b[0] if wall in ("north", "south") else b[1]) + float(op.get("offset", 0.5)) * span
    tgt = norm_label(op.get("target") or op.get("target_room") or "")
    pad = 0.02 if op.get("kind") == "entry_door" else min(0.35, ow * 0.45)
    cands = []
    for other in rooms:
        if other["id"] == room["id"]:
            continue
        ob = other["_bounds"]
        if wall == "north" and abs(b[1] - ob[3]) <= 0.05 and min(b[2], ob[2]) - max(b[0], ob[0]) >= 0.4:
            lo, hi = max(b[0], ob[0]), min(b[2], ob[2])
        elif wall == "south" and abs(b[3] - ob[1]) <= 0.05 and min(b[2], ob[2]) - max(b[0], ob[0]) >= 0.4:
            lo, hi = max(b[0], ob[0]), min(b[2], ob[2])
        elif wall == "west" and abs(b[0] - ob[2]) <= 0.05 and min(b[3], ob[3]) - max(b[1], ob[1]) >= 0.4:
            lo, hi = max(b[1], ob[1]), min(b[3], ob[3])
        elif wall == "east" and abs(b[2] - ob[0]) <= 0.05 and min(b[3], ob[3]) - max(b[1], ob[1]) >= 0.4:
            lo, hi = max(b[1], ob[1]), min(b[3], ob[3])
        else:
            continue
        if lo - pad <= c <= hi + pad:
            out_dist = 0.0 if (lo <= c <= hi) else min(abs(c - lo), abs(c - hi))
            name_match = 0 if (tgt and tgt == norm_label(other["name"])) else 1
            mid_dist = abs(c - 0.5 * (lo + hi))
            cands.append((out_dist > 0.02, name_match, out_dist, mid_dist, other))
    if not cands:
        return None
    cands.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
    return cands[0][4]


def _on_outer_boundary(room: dict, rooms: list, wall: str, op: dict = None) -> bool:
    b = room["_bounds"]
    hx1, hx2 = min(r["_bounds"][0] for r in rooms), max(r["_bounds"][2] for r in rooms)
    hz1, hz2 = min(r["_bounds"][1] for r in rooms), max(r["_bounds"][3] for r in rooms)
    tol = 0.08
    if {"north": abs(b[1] - hz1) <= tol, "south": abs(b[3] - hz2) <= tol,
            "west": abs(b[0] - hx1) <= tol, "east": abs(b[2] - hx2) <= tol}[wall]:
        return True
    # 凹字形/L形建筑外墙：若该门/窗所在外侧 1.5m 内没有任何其它房间遮挡，也是合法外墙
    span = (b[2] - b[0]) if wall in ("north", "south") else (b[3] - b[1])
    c = (b[0] if wall in ("north", "south") else b[1]) + float((op or {}).get("offset", 0.5)) * span
    probe = {
        "north": [c - 0.15, b[1] - 1.5, c + 0.15, b[1] - 0.05],
        "south": [c - 0.15, b[3] + 0.05, c + 0.15, b[3] + 1.5],
        "west": [b[0] - 1.5, c - 0.15, b[0] - 0.05, c + 0.15],
        "east": [b[2] + 0.05, c - 0.15, b[2] + 1.5, c + 0.15],
    }[wall]
    for other in rooms:
        if other["id"] == room["id"]:
            continue
        ox, oz = rect_overlap(probe, other["_bounds"])
        if ox > 0.05 and oz > 0.05:
            return False
    return True


def _cluster(vals: list, tol: float) -> dict:
    if not vals:
        return {}
    uniq = sorted(set(round(v, 4) for v in vals))
    groups = [[uniq[0]]]
    for v in uniq[1:]:
        if v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    mapping = {}
    for g in groups:
        mv = sum(g) / len(g)
        for v in g:
            mapping[v] = mv
    return mapping


def snap_geometry(rooms: list) -> list:
    """只看几何：共享墙聚类 + 最小尺寸 + 重叠裁剪。不碰门洞。"""
    xmap = _cluster([r["_bounds"][0] for r in rooms] + [r["_bounds"][2] for r in rooms], 0.12)
    zmap = _cluster([r["_bounds"][1] for r in rooms] + [r["_bounds"][3] for r in rooms], 0.12)
    for r in rooms:
        b = r["_bounds"]
        b[0], b[2] = xmap[round(b[0], 4)], xmap[round(b[2], 4)]
        b[1], b[3] = zmap[round(b[1], 4)], zmap[round(b[3], 4)]
        if b[2] - b[0] < 0.6:
            b[2] = b[0] + 0.6
        if b[3] - b[1] < 0.6:
            b[3] = b[1] + 0.6
        _refresh(r)
    _resolve_overlaps(rooms)
    return rooms


def _shared_span(room: dict, other: dict, wall: str):
    """两间房在共享墙上的重叠区间（沿墙轴的坐标）。"""
    b, ob = room["_bounds"], other["_bounds"]
    if wall in ("north", "south"):
        lo, hi = max(b[0], ob[0]), min(b[2], ob[2])
    else:
        lo, hi = max(b[1], ob[1]), min(b[3], ob[3])
    return (lo, hi) if hi - lo > 0.3 else None


def _portal_world_xy(room: dict, op: dict) -> tuple:
    b = room["_bounds"]
    wall = op["wall"]
    off = float(op.get("offset", 0.5))
    if wall == "north":
        return (b[0] + off * (b[2] - b[0]), b[1])
    if wall == "south":
        return (b[0] + off * (b[2] - b[0]), b[3])
    if wall == "west":
        return (b[0], b[1] + off * (b[3] - b[1]))
    return (b[2], b[1] + off * (b[3] - b[1]))


def _opp_wall(wall: str) -> str:
    return {"north": "south", "south": "north", "west": "east", "east": "west"}[wall]


def _preserve_connecting_alcove(rooms: list) -> int:
    """若某房间（如厨房/卫生间）门开向内部空隙，且该空隙被客厅/卫生间/厨房围合形成干湿分离过道，则自动补全过道空间。"""
    added = 0
    for r in list(rooms):
        b = r["_bounds"]
        for op in r.get("openings") or []:
            if op.get("kind") not in ("door", "sliding_door", "open_arch", "open_passage"):
                continue
            wall = op["wall"]
            if _find_neighbor(r, rooms, wall, op) or _on_outer_boundary(r, rooms, wall):
                continue
            if wall == "north":
                cands = [o["_bounds"][3] for o in rooms if o["id"] != r["id"]
                         and o["_bounds"][3] < b[1] - 0.35
                         and min(b[2], o["_bounds"][2]) - max(b[0], o["_bounds"][0]) >= 0.5]
                if not cands:
                    continue
                top_z = max(cands)
                if not (0.55 <= b[1] - top_z <= 2.1):
                    continue
                cand_b = [b[0], top_z, b[2], b[1]]
            elif wall == "south":
                cands = [o["_bounds"][1] for o in rooms if o["id"] != r["id"]
                         and o["_bounds"][1] > b[3] + 0.35
                         and min(b[2], o["_bounds"][2]) - max(b[0], o["_bounds"][0]) >= 0.5]
                if not cands:
                    continue
                bot_z = min(cands)
                if not (0.55 <= bot_z - b[3] <= 2.1):
                    continue
                cand_b = [b[0], b[3], b[2], bot_z]
            else:
                continue
            if any(rect_overlap(cand_b, o["_bounds"])[0] > 0.05 and rect_overlap(cand_b, o["_bounds"])[1] > 0.05 for o in rooms):
                continue
            touch_side = None
            for o in rooms:
                ob = o["_bounds"]
                if min(cand_b[3], ob[3]) - max(cand_b[1], ob[1]) < 0.4:
                    continue
                if abs(cand_b[0] - ob[2]) <= 0.15:
                    touch_side = ("west", o)
                    break
                if abs(cand_b[2] - ob[0]) <= 0.15:
                    touch_side = ("east", o)
                    break
            if not touch_side:
                continue
            alc_id = f"rm_{uuid.uuid4().hex[:8]}"
            alc = {
                "id": alc_id, "name": "过道", "category": "entry",
                "icon": CATEGORY_ICON["entry"],
                "rel_x": 0.0, "rel_z": 0.0, "width_m": 1.0, "depth_m": 1.0,
                "window_dir": "NONE",
                "openings": [
                    {"kind": "open_passage", "wall": touch_side[0], "offset": 0.5,
                     "width_m": round(clamp((cand_b[3] - cand_b[1]) * 0.82, 0.75, 1.4), 2),
                     "target": touch_side[1]["name"], "render_leaf": False},
                ],
                "photos": [], "scene_spec": None, "bbox_1000": None,
                "_bounds": cand_b,
            }
            _refresh(alc)
            rooms.append(alc)
            added += 1
    return added


def validate_rooms(rooms: list) -> tuple:
    """门洞落点/连通性/互惠开洞/单侧渲染门扇/零重叠最终校验。"""
    warnings = []
    _preserve_connecting_alcove(rooms)
    _resolve_overlaps(rooms, eps=0.002)

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            ox, oz = rect_overlap(rooms[i]["_bounds"], rooms[j]["_bounds"])
            if ox > 0.01 and oz > 0.01:
                warnings.append(f"房间「{rooms[i]['name']}」与「{rooms[j]['name']}」仍存在 {ox:.2f}×{oz:.2f}m 重叠")

    for r in rooms:
        for op in r["openings"]:
            nb = _find_neighbor(r, rooms, op["wall"], op)
            if op["kind"] in ("door", "sliding_door", "open_arch", "open_passage", "entry_door"):
                if nb:
                    op["target"] = nb["name"]
                    # 把门洞夹进共享墙段，保证门真的开在两面墙都存在的区间
                    span = _shared_span(r, nb, op["wall"])
                    if span and op["kind"] not in ("open_arch", "open_passage"):
                        b = r["_bounds"]
                        wall_start = b[0] if op["wall"] in ("north", "south") else b[1]
                        wall_len = (b[2] - b[0]) if op["wall"] in ("north", "south") else (b[3] - b[1])
                        w = float(op.get("width_m") or 0.9)
                        lo = max(span[0], wall_start) + w / 2
                        hi = min(span[1], wall_start + wall_len) - w / 2
                        if lo <= hi:
                            c = wall_start + float(op.get("offset", 0.5)) * wall_len
                            c = min(max(c, lo), hi)
                            op["offset"] = round(max(0.05, min(0.95, (c - wall_start) / max(0.1, wall_len))), 3)
                elif not _on_outer_boundary(r, rooms, op["wall"], op):
                    warnings.append(f"{r['name']} 的{op['wall']}墙有{op['kind']}，但该侧既无相邻房间也不在外墙上，请复核")
                if op["kind"] == "entry_door" and not _on_outer_boundary(r, rooms, op["wall"], op):
                    warnings.append(f"{r['name']} 的入户门不在外墙，请复核")
            elif nb:
                warnings.append(f"{r['name']} 的{op['wall']}墙窗户面向相邻房间「{nb['name']}」，已按内窗保留")

    # 建筑规范：若卫生间与厨房之间没有过道时被误判直开门，且另有门洞时，移除跨区无效直开门
    drop = set()
    for r in rooms:
        for op in r.get("openings") or []:
            if op.get("kind") not in ("door", "sliding_door", "open_arch"):
                continue
            nb = _find_neighbor(r, rooms, op["wall"], op)
            if nb and {r.get("category"), nb.get("category")} == {"bath", "kitchen"}:
                drop.add((r["id"], id(op)))
    if drop:
        for r in rooms:
            other_doors = [o for o in r.get("openings") or []
                           if (r["id"], id(o)) not in drop and o.get("kind") in ("door", "sliding_door", "open_arch", "open_passage")]
            if other_doors:
                r["openings"] = [o for o in r.get("openings") or [] if (r["id"], id(o)) not in drop]
        warnings.append("检测到卫生间与厨房直连门洞，已清理跨区无效门洞")

    # 互惠门洞补洞：若房间 A 的共享墙有门通往房间 B，确保 B 的对侧墙也有对应开口（render_leaf=False），
    # 防止 B 建出一面实心墙把 A 的门堵死！
    for r in rooms:
        for op in list(r.get("openings") or []):
            if op.get("kind") not in ("door", "sliding_door", "open_arch", "open_passage"):
                continue
            nb = _find_neighbor(r, rooms, op["wall"], op)
            if not nb:
                continue
            opp_w = _opp_wall(op["wall"])
            wx, wz = _portal_world_xy(r, op)
            has_match = False
            for nop in nb.get("openings") or []:
                if nop.get("wall") != opp_w or nop.get("kind") == "window":
                    continue
                nwx, nwz = _portal_world_xy(nb, nop)
                if math.hypot(wx - nwx, wz - nwz) <= 0.50:
                    has_match = True
                    break
            if not has_match:
                nb_b = nb["_bounds"]
                nb_start = nb_b[0] if opp_w in ("north", "south") else nb_b[1]
                nb_len = (nb_b[2] - nb_b[0]) if opp_w in ("north", "south") else (nb_b[3] - nb_b[1])
                coord = wx if opp_w in ("north", "south") else wz
                opp_off = round(clamp((coord - nb_start) / max(0.1, nb_len), 0.06, 0.94), 3)
                nb.setdefault("openings", []).append({
                    "kind": op["kind"],
                    "wall": opp_w,
                    "offset": opp_off,
                    "width_m": op.get("width_m", 0.9),
                    "target": r["name"],
                    "render_leaf": False,
                })

    # 同墙同位置门洞去重（防止同一扇门被重复识别为 entry_door + door 或双门）
    for r in rooms:
        ops = r.get("openings") or []
        kept = []
        for op in ops:
            kind = op.get("kind")
            if kind not in ("door", "sliding_door", "open_arch", "entry_door", "open_passage"):
                kept.append(op)
                continue
            wx, wz = _portal_world_xy(r, op)
            dup = False
            for k in kept:
                if k.get("wall") != op.get("wall") or k.get("kind") == "window":
                    continue
                kx, kz = _portal_world_xy(r, k)
                if math.hypot(wx - kx, wz - kz) <= 0.45:
                    dup = True
                    break
            if not dup:
                kept.append(op)
        r["openings"] = kept

    # 单主权共享门契约（Single-Owner Shared Portal）：
    # 对任意共享墙门/推拉门/垭口，按世界坐标去重，严格只允许一侧 render_leaf=True，对侧强制 render_leaf=False！
    owned_portals = []
    for r in rooms:
        for op in r.get("openings") or []:
            kind = op.get("kind")
            if kind not in ("door", "sliding_door", "open_arch", "entry_door"):
                continue
            nb = _find_neighbor(r, rooms, op["wall"], op)
            if not nb:
                op["render_leaf"] = True
                continue
            wx, wz = _portal_world_xy(r, op)
            if any(math.hypot(wx - px, wz - pz) <= 0.45 for px, pz in owned_portals):
                op["render_leaf"] = False
            else:
                op["render_leaf"] = True
                owned_portals.append((wx, wz))

    hx1, hx2 = min(r["_bounds"][0] for r in rooms), max(r["_bounds"][2] for r in rooms)
    hz1, hz2 = min(r["_bounds"][1] for r in rooms), max(r["_bounds"][3] for r in rooms)
    cover = sum(r["width_m"] * r["depth_m"] for r in rooms) / max(0.1, (hx2 - hx1) * (hz2 - hz1))
    if cover < 0.55:
        warnings.append(f"房间总面积为外接矩形的 {cover:.0%}，可能有漏识别或 L 形公共区域，请复核")

    for r in rooms:
        r.pop("_bounds", None)
    return rooms, warnings


def snap_and_validate(rooms: list) -> tuple:
    """兼容旧调用：先几何后校验。"""
    rooms = snap_geometry(rooms)
    return validate_rooms(rooms)


# --------------------------------------------------------------------------- 二审保守合并


def _bbox_iou(a: list, b: list) -> float:
    ox, oz = rect_overlap(a, b)
    inter = ox * oz
    if inter <= 0:
        return 0.0
    aa = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + ab - inter)


def _house_bbox(rooms: list) -> list:
    return [min(r["bbox_1000"][0] for r in rooms), min(r["bbox_1000"][1] for r in rooms),
            max(r["bbox_1000"][2] for r in rooms), max(r["bbox_1000"][3] for r in rooms)]


def audit_layout_compatible(p1: list, p2: list) -> bool:
    if not p1 or not p2:
        return False
    b1, b2 = _house_bbox(p1), _house_bbox(p2)
    w1, h1 = max(1.0, b1[2] - b1[0]), max(1.0, b1[3] - b1[1])
    w2, h2 = max(1.0, b2[2] - b2[0]), max(1.0, b2[3] - b2[1])
    return (0.70 <= (w2 / h2) / (w1 / h1) <= 1.40) and (0.65 <= (w2 * h2) / (w1 * h1) <= 1.55)


def conservative_merge(p1: list, p2: list) -> tuple:
    by_name = {r["name"]: r for r in p2}
    out, used = [], set()
    accepted = rejected = 0
    for r1 in p1:
        r2 = by_name.get(r1["name"])
        if not r2:
            out.append(r1)
            continue
        used.add(r2["name"])
        a1 = max(1.0, (r1["bbox_1000"][2] - r1["bbox_1000"][0]) * (r1["bbox_1000"][3] - r1["bbox_1000"][1]))
        a2 = max(1.0, (r2["bbox_1000"][2] - r2["bbox_1000"][0]) * (r2["bbox_1000"][3] - r2["bbox_1000"][1]))
        if _bbox_iou(r1["bbox_1000"], r2["bbox_1000"]) >= 0.65 and 0.70 <= a2 / a1 <= 1.40:
            merged = dict(r1)
            merged["bbox_1000"] = r2["bbox_1000"]
            for k in ("est_width_m", "est_depth_m"):
                if r2.get(k):
                    merged[k] = r2[k]
            if r2.get("window_dir") and r2["window_dir"] != "NONE":
                merged["window_dir"] = r2["window_dir"]
            if len(r2.get("openings") or []) >= len(r1.get("openings") or []):
                merged["openings"] = r2["openings"]
            out.append(merged)
            accepted += 1
        else:
            out.append(r1)
            rejected += 1
    new_names = [r["name"] for r in p2 if r["name"] not in used]
    return out, accepted, rejected, new_names


# --------------------------------------------------------------------------- 主流程


def attach_existing(rooms: list, old_rooms: list) -> list:
    old_by_name = {str(r.get("name") or ""): r for r in (old_rooms or [])}
    for r in rooms:
        old = old_by_name.get(r["name"])
        if not old:
            for k, v in old_by_name.items():
                if k and (k in r["name"] or r["name"] in k):
                    old = v
                    break
        if old:
            r["id"] = old.get("id") or r["id"]
            r["photos"] = old.get("photos") or []
            if old.get("scene_spec"):
                r["scene_spec"] = old["scene_spec"]
    return rooms


def run_floorplan_pipeline(fp_abs: str, pdir: str, old_rooms: list, proj: dict, do_audit: bool = True) -> tuple:
    img_w, img_h = image_size(fp_abs)
    trace = {}

    draft = parse_floorplan_draft(fp_abs)
    trace["pass1"] = draft
    log_stage(proj, "floorplan.pass1", "ok",
              f"识别 {len(draft.get('rooms') or [])} 个空间，面积 {draft.get('estimated_area_m2')}m²",
              model=(draft.get("_vlm") or {}).get("model"), elapsed=(draft.get("_vlm") or {}).get("elapsed_s"))

    rooms_raw = [c for c in (_clean_room(x, i) for i, x in enumerate(draft.get("rooms") or [])) if c]
    if not rooms_raw:
        raise RuntimeError("户型图第一轮未识别到任何房间")

    audit = None
    if do_audit:
        stem = os.path.splitext(os.path.basename(fp_abs))[0]
        overlay_abs = os.path.join(pdir, f"debug/{stem}_overlay_pass1.jpg")
        draw_floorplan_overlay(fp_abs, rooms_raw, overlay_abs)
        try:
            audit = audit_floorplan(fp_abs, overlay_abs, draft)
            trace["pass2"] = audit
            audit_rooms = [c for c in (_clean_room(x, i) for i, x in enumerate(audit.get("rooms") or [])) if c]
            if not audit_rooms:
                log_stage(proj, "floorplan.audit", "warn", "二审返回空房间列表，沿用第一轮")
            elif not audit_layout_compatible(rooms_raw, audit_rooms):
                log_stage(proj, "floorplan.audit", "warn",
                          "二审整体布局与一审不同构（疑似旋转/缩放），已整体保留一审坐标")
                pipeline_warning(proj, "floorplan.audit", "二审把户型整体转置/缩放了，已丢弃其坐标，只采纳 pass1")
            else:
                rooms_raw, acc, rej, new_names = conservative_merge(rooms_raw, audit_rooms)
                log_stage(proj, "floorplan.audit", "ok",
                          f"二审后 {len(rooms_raw)} 个空间（采纳 {acc} 处、保留一审 {rej} 处）",
                          notes=(audit.get("audit_notes") or [])[:12],
                          model=(audit.get("_vlm") or {}).get("model"))
                if new_names:
                    pipeline_warning(proj, "floorplan.audit",
                                     f"二审新增了 {new_names}，已记录但不自动采纳（防幻觉空间）")
        except Exception as e:  # noqa: BLE001
            log_stage(proj, "floorplan.audit", "warn", f"二审失败，沿用第一轮：{e}")
        draw_floorplan_overlay(fp_abs, rooms_raw, os.path.join(pdir, f"debug/{stem}_overlay_final.jpg"))

    north = str(draft.get("north_direction") or "up").lower()
    if north == "none":
        north = "up"
    # 保持 3D 平面坐标系与户型图视觉朝向一致（图上为北 -Z、图右为东 +X），
    # 避免只旋转坐标而不旋转原图导致像素级贴边/门洞缺口对齐/局部裁图全部错位；
    # 指北针朝向仅作为元数据保留在 trace["north_direction"]。
    trace["north_direction"] = north

    # 1) 像素级：bbox 先吸到强墙线（含重叠房间共用墙），再裁白边
    before = [r["bbox_1000"][:] for r in rooms_raw]
    rooms_raw = refine.snap_rooms_to_walls(rooms_raw, fp_abs)
    rooms_raw = refine.trim_rooms_to_content(rooms_raw, fp_abs)
    trimmed = sum(1 for a, b in zip(before, rooms_raw) if a != b["bbox_1000"])
    if trimmed:
        log_stage(proj, "floorplan.trim", "ok", f"{trimmed} 个房间的 bbox 按墙体/地面前景像素收边")
    draw_floorplan_overlay(fp_abs, rooms_raw, os.path.join(pdir, f"debug/{os.path.splitext(os.path.basename(fp_abs))[0]}_overlay_trim.jpg"))

    est_area = float((audit or draft).get("estimated_area_m2") or 0)
    scale, scale_info = calibrate_scale(rooms_raw, img_w, img_h, est_area)
    trace["scale"] = scale_info
    if scale_info["choice"].startswith("room_dims"):
        pipeline_warning(proj, "floorplan.scale",
                         f"面积尺度与房间尺寸尺度相差 {scale_info.get('ratio_b_over_a')} 倍，已采信房间尺寸")

    # 2) 像素级：沿墙线缺口锁死每扇门/窗的 offset 与宽度
    rooms_raw = refine.apply_wall_gaps(rooms_raw, fp_abs, scale, img_w, img_h)
    log_stage(proj, "floorplan.door_gaps", "ok",
              "墙线缺口已用于校正 "
              + "；".join(f"{r['name']}:{len(r.get('openings') or [])}洞" for r in rooms_raw))

    xs = [r["bbox_1000"][0] for r in rooms_raw] + [r["bbox_1000"][2] for r in rooms_raw]
    zs = [r["bbox_1000"][1] for r in rooms_raw] + [r["bbox_1000"][3] for r in rooms_raw]
    cx_px, cz_px = (min(xs) + max(xs)) / 2000.0 * img_w, (min(zs) + max(zs)) / 2000.0 * img_h

    rooms = [_build_room(item, f"rm_{uuid.uuid4().hex[:8]}", cx_px, cz_px, scale, img_w, img_h)
             for item in rooms_raw]

    # 3) 先聚类/裁重叠，最后才做门洞合缝——顺序反了会把合好的缝再裁开
    rooms = snap_geometry(rooms)
    snapped = _snap_door_walls(rooms)
    if snapped:
        log_stage(proj, "floorplan.door_snap", "ok", f"按门洞连通关系合上 {snapped} 处墙缝")

    rooms, warnings = validate_rooms(rooms)
    for w in warnings:
        pipeline_warning(proj, "floorplan.validate", w)
    trace["warnings"] = warnings

    # 用最终尺寸反推 bbox 重画审核叠加图，避免「图上框和 3D 实际不一致」
    final_debug = []
    for r in rooms:
        x1 = (r["rel_x"] - r["width_m"] / 2) / scale + cx_px
        x2 = (r["rel_x"] + r["width_m"] / 2) / scale + cx_px
        z1 = (r["rel_z"] - r["depth_m"] / 2) / scale + cz_px
        z2 = (r["rel_z"] + r["depth_m"] / 2) / scale + cz_px
        final_debug.append({"name": r["name"], "bbox_1000": [
            round(x1 / img_w * 1000, 1), round(z1 / img_h * 1000, 1),
            round(x2 / img_w * 1000, 1), round(z2 / img_h * 1000, 1)]})
    draw_floorplan_overlay(fp_abs, final_debug, os.path.join(
        pdir, f"debug/{os.path.splitext(os.path.basename(fp_abs))[0]}_overlay_final.jpg"))

    rooms = attach_existing(rooms, old_rooms)
    summary = str((audit or draft).get("house_summary") or f"已识别 {len(rooms)} 个空间")
    trace["rooms"] = [{k: v for k, v in r.items() if k != "scene_spec"} for r in rooms]
    log_stage(proj, "floorplan.done", "ok",
              f"{summary}；{len(rooms)} 个空间，尺度 {scale:.4f} m/px（{scale_info['choice']}）",
              rooms=len(rooms), warnings=len(warnings))
    return summary, rooms, trace
