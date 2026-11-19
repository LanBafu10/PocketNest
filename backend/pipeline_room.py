#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""房间/家具解析流水线（PocketNest v3 Stage 2）。

流程：
  crop   render_annotated_room_crop()   生成标有 N/E/S/W 与门窗锚点的房间局部户型图
  orient calibrate_photo_orientations() 结合局部户型图对每张实景照片做机位定标
  pass1  analyze_room_pass1()           多锚点空间图谱 + 墙面结构反哺 + 父子依附识别
  merge  merge_multiview()              跨视角去重（同帧分离同类座椅绝不误合并）
  pass2  audit_room_pass2()             结合局部户型图 + 实景照片逐件复核修正
  solve  solve_room_layout()            主家具优先（禁扭转床位）+ 阳台子区隔离 + 父子刚体组合包
"""

from __future__ import annotations

import math
import os
import uuid

from PIL import Image, ImageDraw, ImageFont

from pipeline_common import (
    AUDIT_VLM_MODEL,
    ROOM_VLM_MODEL,
    call_vlm_json,
    clamp,
    log_stage,
    median,
    norm_label,
    pipeline_warning,
)

TYPE_DEFAULT_DIMS = {
    "bed": (1.58, 0.90, 2.02),
    "wardrobe": (1.42, 2.02, 0.58),
    "desk": (1.18, 1.38, 0.58),
    "chair": (0.48, 0.84, 0.48),
    "nightstand": (0.46, 0.54, 0.44),
    "sofa": (2.10, 0.84, 0.88),
    "tv_stand": (1.20, 0.65, 0.38),
    "tv": (1.10, 0.66, 0.12),
    "coffee_table": (1.00, 0.42, 0.55),
    "side_table": (0.52, 0.56, 0.52),
    "dining_table": (1.35, 0.76, 0.82),
    "plant": (0.45, 0.95, 0.45),
    "kitchen_counter": (2.60, 0.86, 0.60),
    "wall_cabinet": (2.10, 0.68, 0.36),
    "cabinet": (0.80, 1.20, 0.44),
    "fridge": (0.76, 1.78, 0.68),
    "bathtub": (1.50, 0.58, 0.72),
    "vanity_sink": (0.88, 0.86, 0.50),
    "toilet": (0.44, 0.78, 0.68),
    "washer": (0.62, 0.86, 0.60),
    "drying_rack": (1.15, 1.50, 0.50),
    "bookshelf": (0.72, 1.92, 0.45),
    "rug": (1.60, 0.03, 1.20),
    "other": (0.70, 0.85, 0.48),
}

ANCHOR_TO_ROT = {
    "north": 0,
    "corner_nw": 0,
    "corner_ne": 0,
    "south": 180,
    "corner_sw": 180,
    "corner_se": 180,
    "west": 90,
    "east": 270,
    "center": 0,
}

# 家具「正面」朝向某面墙时的 rotation.y（局部 +Z 为正面）
FACING_TO_ROT = {"north": 180, "south": 0, "west": 270, "east": 90}

ATTACH_SLOTS = (
    "around_table",
    "front_of_desk",
    "above_stand",
    "wall_high_above",
    "beside_headboard",
    "beside_headboard_left",
    "beside_headboard_right",
)


# --------------------------------------------------------------------------- 局部户型标注图


def _estimate_room_bbox_1000(room: dict, all_rooms: list) -> list | None:
    bb = room.get("bbox_1000")
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return [float(v) for v in bb]
    # 若是自动补全的过道等无 bbox_1000 空间，用有 bbox_1000 的房间反推仿射比例
    ref = [r for r in all_rooms if isinstance(r.get("bbox_1000"), (list, tuple)) and len(r["bbox_1000"]) == 4]
    if not ref:
        return None
    r0 = ref[0]
    b0 = [float(v) for v in r0["bbox_1000"]]
    rw0, rd0 = max(0.5, float(r0.get("width_m") or 1.0)), max(0.5, float(r0.get("depth_m") or 1.0))
    sx = (b0[2] - b0[0]) / rw0
    sz = (b0[3] - b0[1]) / rd0
    cx0, cz0 = (b0[0] + b0[2]) / 2.0, (b0[1] + b0[3]) / 2.0
    dx = float(room.get("rel_x", 0)) - float(r0.get("rel_x", 0))
    dz = float(room.get("rel_z", 0)) - float(r0.get("rel_z", 0))
    hw = float(room.get("width_m", 1.0)) * sx / 2.0
    hd = float(room.get("depth_m", 1.0)) * sz / 2.0
    cx, cz = cx0 + dx * sx, cz0 + dz * sz
    return [clamp(cx - hw, 0, 1000), clamp(cz - hd, 0, 1000), clamp(cx + hw, 0, 1000), clamp(cz + hd, 0, 1000)]


def render_annotated_room_crop(fp_abs: str, room: dict, all_rooms: list, out_path: str) -> str | None:
    """裁出围绕当前房间的局部户型图，标注 N/E/S/W 方位与全部门窗开口位置，供 VLM 对照。"""
    if not fp_abs or not os.path.isfile(fp_abs):
        return None
    bb = _estimate_room_bbox_1000(room, all_rooms)
    with Image.open(fp_abs) as im:
        im = im.convert("RGB")
        W, H = im.size
        draw = ImageDraw.Draw(im)

        font_lg, font_sm = None, None
        for fpath in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if os.path.isfile(fpath):
                try:
                    font_lg = ImageFont.truetype(fpath, max(18, int(min(W, H) * 0.024)))
                    font_sm = ImageFont.truetype(fpath, max(13, int(min(W, H) * 0.016)))
                    break
                except Exception:
                    pass

        if bb:
            x1, y1, x2, y2 = [bb[0] / 1000.0 * W, bb[1] / 1000.0 * H, bb[2] / 1000.0 * W, bb[3] / 1000.0 * H]
            # 高亮当前房间边界
            for w in range(4):
                draw.rectangle([x1 - w, y1 - w, x2 + w, y2 + w], outline=(53, 139, 132))
            # 四面墙 N / S / W / E 徽章
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            for txt, tx, ty in (
                ("N (North)", mx, max(16.0, y1 - 14)),
                ("S (South)", mx, min(H - 16.0, y2 + 14)),
                ("W (West)", max(38.0, x1 - 32), my),
                ("E (East)", min(W - 38.0, x2 + 32), my),
            ):
                tw = len(txt) * 8 + 10
                draw.rectangle([tx - tw / 2, ty - 11, tx + tw / 2, ty + 11], fill=(29, 53, 87))
                draw.text((tx - tw / 2 + 5, ty - 8), txt, fill=(255, 255, 255), font=font_sm)

            # 标注每个已知门/窗/垭口位置
            kind_tag = {"door": "DOOR", "sliding_door": "SLIDE", "open_arch": "ARCH",
                        "open_passage": "PASS", "entry_door": "ENTRY", "window": "WIN"}
            for op in room.get("openings") or []:
                wall = op.get("wall")
                off = float(op.get("offset", 0.5))
                if wall == "north":
                    px, py = x1 + off * (x2 - x1), y1
                elif wall == "south":
                    px, py = x1 + off * (x2 - x1), y2
                elif wall == "west":
                    px, py = x1, y1 + off * (y2 - y1)
                elif wall == "east":
                    px, py = x2, y1 + off * (y2 - y1)
                else:
                    continue
                is_win = op.get("kind") == "window"
                col = (47, 128, 237) if is_win else (230, 57, 70)
                draw.ellipse([px - 7, py - 7, px + 7, py + 7], fill=col, outline=(255, 255, 255))
                tag = kind_tag.get(op.get("kind"), "DOOR")
                draw.rectangle([px + 8, py - 9, px + 8 + len(tag) * 9 + 6, py + 9], fill=col)
                draw.text((px + 11, py - 7), tag, fill=(255, 255, 255), font=font_sm)

            # 留出足够邻域视野进行裁剪
            pad_x = max(W * 0.16, (x2 - x1) * 0.65)
            pad_y = max(H * 0.16, (y2 - y1) * 0.65)
            cx1 = int(max(0, x1 - pad_x))
            cy1 = int(max(0, y1 - pad_y))
            cx2 = int(min(W, x2 + pad_x))
            cy2 = int(min(H, y2 + pad_y))
            if cx2 - cx1 >= 120 and cy2 - cy1 >= 120:
                im = im.crop((cx1, cy1, cx2, cy2))

        im.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path, format="JPEG", quality=90)
    return out_path


# --------------------------------------------------------------------------- 机位定标


def calibrate_photo_orientations(
    proj: dict, room: dict, fp_abs: str, photo_abs: list, crop_map_abs: str | None = None
) -> list:
    """逐张照片做「以门窗为锚点」的机位定标。"""
    map_img = crop_map_abs if (crop_map_abs and os.path.isfile(crop_map_abs)) else fp_abs
    if not map_img or not photo_abs:
        return []
    walls = describe_walls(room, proj.get("rooms", []))
    prompt = f"""你是一位室内摄影测量员。第 0 张图是本房间【{room['name']}】及周边邻接空间的局部户型图（已用 N/S/W/E 标注四面墙方向，红点=门洞，蓝点=窗洞）。第 1..{len(photo_abs)} 张是同一个房间【{room['name']}】的实景照片，photo_index 依次为 0..{len(photo_abs) - 1}。

【户型图已锁定的四面墙与门窗】
{walls}

【任务】对每一张照片单独做机位定标，不要跨照片混用：
1. 先在画面里找可见的门、门套、垭口、窗户，对照第 0 张局部户型图与上面锁定文字，判断它属于哪面墙、偏左/居中/偏右、通向哪里。
2. 由此推出这张照片：相机正前方是哪面墙（front_wall）、画面左侧是哪面墙（left_wall）、画面右侧是哪面墙（right_wall）、镜头背后是哪面墙（back_wall）。
3. 几何铁律：
   - 镜头朝北：前=北，左=西，右=东；
   - 镜头朝南：前=南，左=东，右=西；
   - 镜头朝西：前=西，左=南，右=北；
   - 镜头朝东：前=东，左=北，右=南。
4. 如果一张照片只看到相邻家具、看不到门窗，可通过与其它照片的共视家具链（如床、衣柜、书桌的相对位置）推导该照片机位；若实在无法确认，front_wall 填 unknown、confidence 填低。

严格输出 JSON（不要多余文字）：
{{
  "photos": [
    {{"photo_index": 0, "front_wall": "north", "left_wall": "west", "right_wall": "east",
      "back_wall": "south", "anchor_evidence": "画面左侧可见北墙的窗，右侧是东墙", "confidence": 0.9}}
  ],
  "notes": []
}}"""
    try:
        res = call_vlm_json(
            prompt,
            [map_img] + photo_abs[:10],
            timeout=180,
            max_images=11,
            model=ROOM_VLM_MODEL,
            max_tokens=16384,
        )
        out = []
        for p in res.get("photos") or []:
            if not isinstance(p, dict):
                continue
            item = {"photo_index": int(p.get("photo_index", len(out)))}
            for k in ("front_wall", "left_wall", "right_wall", "back_wall"):
                v = str(p.get(k) or "unknown").lower()
                item[k] = v if v in ("north", "south", "west", "east") else "unknown"
            item["anchor_evidence"] = str(p.get("anchor_evidence") or "")[:160]
            try:
                item["confidence"] = round(clamp(float(p.get("confidence", 0.6)), 0, 1), 2)
            except Exception:
                item["confidence"] = 0.6
            out.append(item)
        return out
    except Exception:
        return []


# --------------------------------------------------------------------------- 辅助工具


def _safe_type(t: str, label: str = "") -> str:
    t = str(t or "other").lower().strip()
    if t in TYPE_DEFAULT_DIMS:
        return t
    if "电视" in label and "柜" not in label:
        return "tv"
    if "晾衣" in label:
        return "drying_rack"
    if "边桌" in label or "床头" in label:
        return "nightstand" if "床头" in label else "side_table"
    return "other"


def _color_rgb(hex_str: str) -> tuple:
    s = str(hex_str or "").lstrip("#")
    if len(s) != 6:
        return (200, 200, 200)
    try:
        return tuple(int(s[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (200, 200, 200)


def _color_dist(a: str, b: str) -> float:
    ra, ga, ba = _color_rgb(a)
    rb, gb, bb = _color_rgb(b)
    return math.sqrt((ra - rb) ** 2 + (ga - gb) ** 2 + (ba - bb) ** 2)


def _bbox_iou_1000(a: list | None, b: list | None) -> float:
    if not (isinstance(a, list) and len(a) == 4 and isinstance(b, list) and len(b) == 4):
        return 0.0
    ax1, ay1, ax2, ay2 = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    bx1, by1, bx2, by2 = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    ab = max(1.0, (bx2 - bx1) * (by2 - by1))
    return inter / (aa + ab - inter)


def crop_furniture_image(src_path: str, bbox_1000: list, out_path: str, pad_ratio: float = 0.08):
    """按 bbox 裁出单品缩略图。"""
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        if isinstance(bbox_1000, (list, tuple)) and len(bbox_1000) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox_1000]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            pad_x = max(8.0, (x2 - x1) * pad_ratio)
            pad_y = max(8.0, (y2 - y1) * pad_ratio)
            left = int(max(0, (x1 - pad_x) / 1000.0 * W))
            top = int(max(0, (y1 - pad_y) / 1000.0 * H))
            right = int(min(W, (x2 + pad_x) / 1000.0 * W))
            bottom = int(min(H, (y2 + pad_y) / 1000.0 * H))
            if right - left >= 48 and bottom - top >= 48:
                im = im.crop((left, top, right, bottom))
        im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path, format="JPEG", quality=92)


def crop_context_image(src_path: str, bbox_1000: list, out_path: str, expand: float = 1.9, min_side: int = 780):
    """给 clean-plate 用的上下文裁图。"""
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        W, H = im.size
        if isinstance(bbox_1000, (list, tuple)) and len(bbox_1000) == 4:
            x1, y1, x2, y2 = [float(v) for v in bbox_1000]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            cx, cy = (x1 + x2) / 2 / 1000.0 * W, (y1 + y2) / 2 / 1000.0 * H
            w = max((x2 - x1) / 1000.0 * W * expand, min_side)
            h = max((y2 - y1) / 1000.0 * H * expand, min_side)
            w, h = min(w, W), min(h, H)
            left = int(max(0, cx - w / 2))
            top = int(max(0, cy - h / 2))
            right = int(min(W, left + w))
            bottom = int(min(H, top + h))
            left = int(max(0, right - w))
            top = int(max(0, bottom - h))
            if right - left >= 96 and bottom - top >= 96:
                im = im.crop((left, top, right, bottom))
        im.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        im.save(out_path, format="JPEG", quality=93)


# --------------------------------------------------------------------------- 墙面描述与提示词


def _wall_neighbors(room: dict, all_rooms: list) -> dict:
    rw, rd = float(room.get("width_m", 3.4)), float(room.get("depth_m", 3.4))
    rx, rz = float(room.get("rel_x", 0.0)), float(room.get("rel_z", 0.0))
    x1, z1, x2, z2 = rx - rw / 2, rz - rd / 2, rx + rw / 2, rz + rd / 2
    out = {"north": [], "south": [], "west": [], "east": []}
    for other in all_rooms:
        if other["id"] == room["id"]:
            continue
        ox, oz = float(other.get("rel_x", 0)), float(other.get("rel_z", 0))
        ow, od = float(other.get("width_m", 0)), float(other.get("depth_m", 0))
        ox1, oz1, ox2, oz2 = ox - ow / 2, oz - od / 2, ox + ow / 2, oz + od / 2
        if abs(z1 - oz2) < 0.16 and min(x2, ox2) - max(x1, ox1) > 0.35:
            out["north"].append(other["name"])
        elif abs(z2 - oz1) < 0.16 and min(x2, ox2) - max(x1, ox1) > 0.35:
            out["south"].append(other["name"])
        elif abs(x1 - ox2) < 0.16 and min(z2, oz2) - max(z1, oz1) > 0.35:
            out["west"].append(other["name"])
        elif abs(x2 - ox1) < 0.16 and min(z2, oz2) - max(z1, oz1) > 0.35:
            out["east"].append(other["name"])
    return out


def describe_walls(room: dict, all_rooms: list) -> str:
    rw, rd = float(room.get("width_m", 3.4)), float(room.get("depth_m", 3.4))
    titles = {
        "north": "北墙 (north / -Z)",
        "south": "南墙 (south / +Z)",
        "west": "西墙 (west / -X)",
        "east": "东墙 (east / +X)",
    }
    kind_cn = {
        "door": "平开房门",
        "sliding_door": "推拉门",
        "open_arch": "无门垭口",
        "open_passage": "敞开过道",
        "entry_door": "入户大门",
        "window": "窗户",
    }
    neigh_map = _wall_neighbors(room, all_rooms)
    lines = []
    for wkey, wtitle in titles.items():
        ops = [o for o in (room.get("openings") or []) if o.get("wall") == wkey]
        desc = []
        for o in ops:
            off = float(o.get("offset", 0.5))
            if wkey in ("north", "south"):
                hint = "偏西侧" if off < 0.38 else ("偏东侧" if off > 0.62 else "居中")
            else:
                hint = "偏北侧" if off < 0.38 else ("偏南侧" if off > 0.62 else "居中")
            tgt = f"(通往{o['target']})" if o.get("target") else ""
            desc.append(f"{hint}有{kind_cn.get(o.get('kind'), '门')}{tgt}")
        neigh = neigh_map[wkey]
        n_str = f"隔壁紧邻【{'、'.join(neigh)}】" if neigh else "建筑外侧墙（可能存在简版户型图漏画的外窗）"
        feat = "；".join(desc) if desc else "户型图暂未标门窗"
        lines.append(f"  - {wtitle}（长 {rw if wkey in ('north', 'south') else rd:.2f}m，{n_str}）：{feat}")
    return "\n".join(lines)


def _pass1_prompt(
    room: dict, walls_desc: str, n_photos: int, orientations: list = None, has_map_img: bool = False
) -> str:
    cn = {"north": "北", "south": "南", "west": "西", "east": "东", "unknown": "未知"}
    if orientations:
        lines = ["【逐张照片的机位定标（已按门窗锚点确认，必须作为翻译画面左右的基准）】"]
        for o in orientations:
            lines.append(
                f"  photo {o.get('photo_index')}: 镜头朝{cn.get(o.get('front_wall'), '未知')}，"
                f"画面左={cn.get(o.get('left_wall'), '未知')}，右={cn.get(o.get('right_wall'), '未知')}，"
                f"背后={cn.get(o.get('back_wall'), '未知')}（锚点：{o.get('anchor_evidence') or '无'}）"
            )
        orient_block = "\n".join(lines)
    else:
        orient_block = "【机位未定标】请结合第 0 张局部户型图上的 N/S/W/E 与门窗位置先判断每张照片镜头朝向。"

    img_header = (
        f"第 0 张图是本房间【{room['name']}】及邻接空间的局部户型图（标有 N/S/W/E 与已知门窗点位）。\n"
        f"第 1..{n_photos} 张图都是房间【{room['name']}】（{room.get('category')}，东西宽 {room.get('width_m')}m × 南北深 {room.get('depth_m')}m）的实景照片，"
        f"注意：下面 JSON 中的 photo_index / best_photo_index / seen_in_photos **一律从 0 开始对应这 {n_photos} 张实景照片**（0..{n_photos - 1}，不含第 0 张户型图）。"
        if has_map_img
        else f"下面 {n_photos} 张图都是房间【{room['name']}】（{room.get('category')}，东西宽 {room.get('width_m')}m × 南北深 {room.get('depth_m')}m）的实景照片，photo_index 为 0..{n_photos - 1}。"
    )

    return f"""你是一位 3D 室内重建的空间图谱测绘师。{img_header}

【户型图已经锁定的四面墙与门窗锚点】
{walls_desc}

{orient_block}

【任务 A：实景墙面结构反哺（wall_features_detected）】
1. **外窗补全（windows）**：仔细看实景照片！如果某面「建筑外侧墙」上在照片里真实存在窗户或窗帘（哪怕简版户型图上漏画了），请记录到 `wall_features_detected.windows`；内墙不要标窗。
2. **连通生活阳台垭口子区识别（subzone_split）**：仔细检查狭长房间某一端是否存在由「原木/木质门套垭口 + 双侧或单侧窗帘」半隔断出的连通阳台区（常见内部摆有洗衣机、晾衣架、阳台柜或大外窗）。若有，`has_connected_balcony = true`，给出 `balcony_end`（north/south/west/east）与纵深占比 `balcony_ratio`（0.20~0.36）。若无，填 false。

【任务 B：多锚点共视链推导家具与父子依附关系】
推导每件家具位置时，必须用「① 门的位置与铰链侧 + ② 窗户/阳台垭口方向 + ③ 跨照片共视邻接链（谁挨着谁、谁在谁对面）」三锚点互证，绝不能凭单张照片左右瞎猜！
硬性规则：
1. 只列长期保留的家具/固定装置；忽略纸箱、散落衣物、塑料袋、餐具等杂物。
2. 同一件家具多张照片出现只输出一条并用 `seen_in_photos` 记录；**多把同类椅子（如 4 把餐椅）、两个床头柜必须逐件分开列**，绝不合并！
3. `zone`：若房间存在连通阳台子区，阳台内物品（洗衣机、晾衣架、阳台柜等）填 `"balcony"`，主区家具（床、衣柜、书桌、电视、沙发等）填 `"main"`；无阳台子区则全填 `"main"`。
4. `placement_mode`：`"wall"`（背贴墙）| `"corner"`（贴两面墙转角）| `"center"`（独立摆放）| `"attached"`（依附父家具）。
5. `wall_anchor`：north | south | west | east | corner_nw | corner_ne | corner_sw | corner_se | center。
6. `wall_slot_order`：同一面墙从西到东（北/南墙）或从北到南（东/西墙）的排列序号（0, 1, 2...）；`along_wall_offset` (-1~1) 为同墙相对位置。
7. **父子依附关系（`parent_ref` 与 `attach_slot`，极其重要）**：
   - 餐椅围绕餐桌：`placement_mode="attached"`, `parent_ref` 填餐桌的 label，`attach_slot="around_table"`；
   - 书桌椅/电脑椅/梳妆椅：`placement_mode="attached"`, `parent_ref` 填书桌的 label，`attach_slot="front_of_desk"`；
   - 电视/显示器：`placement_mode="attached"`, 若同墙有电视柜则 `parent_ref` 填电视柜 label、`attach_slot="above_stand"`（摆在柜上）或 `"wall_high_above"`（壁挂在电视柜上方）；若无电视柜则 `mount="wall_high"`（壁挂贴墙，绝不落地！）；
   - 床头柜：`placement_mode="attached"`, `parent_ref` 填床的 label，`attach_slot="beside_headboard"`（紧贴床头同侧墙）。
   - 独立主家具：`parent_ref: null, attach_slot: null`。
8. 尺寸单位米：`[frontal_width_m 正面宽, height_m 高, depth_m 进深]`。
9. `appearance`：20~40 字描述颜色、材质、造型，不提背景与杂物。
10. `facing`：`room_center` / `table` / `sofa` / `bed` / `tv` / `north|south|west|east`，`facing_target` 填对应目标 label。
11. 透过门看到的隔壁房间家具（如从厨房门看到干区洗手台）不要算进本房间！

严格输出纯 JSON：
{{
  "wall_hex": "#F2EEE7",
  "floor_hex": "#C9A176",
  "align_summary": "一句话总结房间布局与多锚点共视链关系",
  "wall_features_detected": {{
    "windows": [
      {{"wall": "north", "offset": 0.5, "width_m": 1.3, "has_curtain": true, "evidence": "照片可见北墙带窗帘外窗"}}
    ],
    "subzone_split": {{
      "has_connected_balcony": false,
      "balcony_end": "south",
      "balcony_ratio": 0.27,
      "divider_kind": "open_arch",
      "has_curtain": true,
      "evidence": ""
    }}
  }},
  "furniture": [
    {{
      "type": "bed | wardrobe | desk | chair | nightstand | sofa | tv_stand | tv | coffee_table | side_table | dining_table | plant | kitchen_counter | fridge | wall_cabinet | cabinet | bathtub | vanity_sink | toilet | washer | drying_rack | bookshelf | rug | other",
      "label": "12 字以内的真实家具名",
      "color_hex": "#CFCBC4",
      "appearance": "浅灰布艺软包，深色木框，床头有弧形靠背",
      "seen_in_photos": [0, 2],
      "best_photo_index": 0,
      "bbox_1000": [120, 280, 860, 900],
      "zone": "main",
      "placement_mode": "wall",
      "wall_anchor": "east",
      "wall_slot_order": 0,
      "along_wall_offset": 0.2,
      "center_z_offset": 0.0,
      "parent_ref": null,
      "attach_slot": null,
      "facing": "west",
      "facing_target": null,
      "mount": "floor",
      "frontal_width_m": 1.50,
      "height_m": 0.90,
      "depth_m": 1.98,
      "is_glass": false,
      "confidence": 0.88
    }}
  ]
}}"""


def _pass2_prompt(
    merged: list,
    walls_desc: str = "",
    orientations: list = None,
    wall_features: dict = None,
    n_photos: int = 0,
    has_map_img: bool = False,
) -> str:
    cn = {"north": "北", "south": "南", "west": "西", "east": "东", "unknown": "未知"}
    orient_txt = ""
    if orientations:
        orient_txt = "【机位定标（复核时按这个翻译画面左右）】\n" + "\n".join(
            f"  photo {o.get('photo_index')}: 镜头朝{cn.get(o.get('front_wall'), '未知')}，"
            f"左={cn.get(o.get('left_wall'), '未知')}，右={cn.get(o.get('right_wall'), '未知')}"
            for o in orientations
        )
    wf_txt = ""
    if wall_features:
        import json
        wf_txt = "【上一轮识别到的墙面结构反哺】\n" + json.dumps(wall_features, ensure_ascii=False)

    img_note = (
        f"第 0 张图是局部户型图，第 1..{n_photos} 张是实景照（photo_index 为 0..{n_photos - 1}）。"
        if has_map_img
        else "以下照片均为同一房间实景照（photo_index 从 0 开始）。"
    )
    return f"""你是 3D 重建的空间图谱复核员。{img_note}

【户型图锁定的四面墙与门窗（复核贴墙方向的唯一基准）】
{walls_desc}

{orient_txt}

{wf_txt}

上一轮已经清点出以下家具：

{_compact_items(merged)}

请逐件复核并对照第 0 张局部户型图和全部实景照片纠偏：
1. exists：这件家具在照片里是否真实存在？不存在（幻觉/隔壁房间透过门看到的家具）填 false。
2. duplicate_of：是否与同房间另一件完全重复（同一只物体在不同照片被输出两次）？注意：**多把餐椅、两个床头柜绝不能判 duplicate**！
3. **三锚点互证 wall_anchor / along_wall_offset / zone / wall_slot_order**：
   - 检查床、衣柜、书桌、沙发、电视、冰箱、洗衣机贴的是哪面真墙！利用门、窗/阳台垭口和照片间共视邻接关系纠偏。
   - 若房间有连通阳台子区，洗衣机/晾衣架必须在 `zone="balcony"`，主卧家具必须在 `zone="main"`。
4. **父子依附复核（parent_ref / attach_slot）**：
   - 餐椅必须依附餐桌 (`attach_slot="around_table"`)；
   - 书桌椅必须依附书桌 (`attach_slot="front_of_desk"`)；
   - 电视必须依附电视柜 (`above_stand` / `wall_high_above`) 或壁挂 (`mount="wall_high"`)；
   - 床头柜依附床 (`beside_headboard`)。
5. **wall_features_detected**：同步复核是否有遗漏的外窗（`windows`）或连通阳台垭口隔断（`subzone_split`）。
6. 如有遗漏的大家具（如餐椅、床头柜、电视、衣帽架等），放进 `missing`。

严格输出纯 JSON：
{{
  "wall_features_detected": {{
    "windows": [],
    "subzone_split": {{"has_connected_balcony": false, "balcony_end": "south", "balcony_ratio": 0.27, "divider_kind": "open_arch", "has_curtain": true}}
  }},
  "corrections": [
    {{"id": "item_xxx", "exists": true, "duplicate_of": null, "zone": "main",
      "placement_mode": "wall", "wall_anchor": "east", "wall_slot_order": 0,
      "along_wall_offset": 0.2, "center_z_offset": 0.0,
      "parent_ref": null, "attach_slot": null,
      "facing": "west", "facing_target": null, "mount": "floor",
      "frontal_width_m": 1.50, "height_m": 0.90, "depth_m": 1.98,
      "best_photo_index": 1, "bbox_1000": [110, 260, 880, 920],
      "label": "可选：修正后的名字", "reason": "简短说明"}}
  ],
  "missing": [],
  "notes": ["整体复核结论"]
}}"""


def _compact_items(items: list) -> str:
    rows = []
    for it in items:
        rows.append(
            "{"
            + ", ".join([
                f'"id":"{it["id"]}"',
                f'"type":"{it["type"]}"',
                f'"label":"{it["label"]}"',
                f'"color_hex":"{it.get("color_hex", "")}"',
                f'"seen_in_photos":{it.get("seen_in_photos") or []}',
                f'"best_photo_index":{it.get("best_photo_index", 0)}',
                f'"bbox_1000":{it.get("bbox_1000")}',
                f'"zone":"{it.get("zone", "main")}"',
                f'"placement_mode":"{it.get("placement_mode", "wall")}"',
                f'"wall_anchor":"{it.get("wall_anchor")}"',
                f'"wall_slot_order":{it.get("wall_slot_order", 0)}',
                f'"along_wall_offset":{it.get("along_wall_offset")}',
                f'"center_z_offset":{it.get("center_z_offset", 0)}',
                f'"parent_ref":{repr(it.get("parent_ref")).replace("None", "null").replace(chr(39), chr(34))}',
                f'"attach_slot":{repr(it.get("attach_slot")).replace("None", "null").replace(chr(39), chr(34))}',
                f'"facing":"{it.get("facing") or ""}"',
                f'"facing_target":"{it.get("facing_target") or ""}"',
                f'"mount":"{it.get("mount", "floor")}"',
                f'"frontal_width_m":{it.get("frontal_width_m")}',
                f'"height_m":{it.get("height_m")}',
                f'"depth_m":{it.get("depth_m")}',
                f'"confidence":{it.get("confidence", 0.6)}',
            ])
            + "}"
        )
    return "[\n" + ",\n".join(rows) + "\n]"


# --------------------------------------------------------------------------- pass1 & 规范化


def _normalize_item(raw: dict, idx: int) -> dict | None:
    if not isinstance(raw, dict):
        return None
    label = str(raw.get("label") or "").strip()[:22]
    ftype = _safe_type(raw.get("type"), label)
    if not label:
        label = ftype
    bb = raw.get("bbox_1000")
    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
        bb = None
    else:
        bb = [round(clamp(float(v), 0, 1000), 1) for v in bb]
        if abs(bb[2] - bb[0]) < 10 or abs(bb[3] - bb[1]) < 10:
            bb = None
    seen = raw.get("seen_in_photos")
    if not isinstance(seen, list):
        seen = [raw.get("best_photo_index", 0)]
    seen = [int(x) for x in seen if isinstance(x, (int, float)) and 0 <= int(x) <= 30]
    try:
        anchor = str(raw.get("wall_anchor") or "north").lower().strip()
    except Exception:
        anchor = "north"
    if anchor not in ANCHOR_TO_ROT:
        anchor = "north"

    def fnum(key, default):
        try:
            return float(raw.get(key) if raw.get(key) is not None else default)
        except Exception:
            return float(default)

    dw, dh, dd = TYPE_DEFAULT_DIMS[ftype]
    zone = str(raw.get("zone") or "main").lower().strip()
    if zone not in ("main", "balcony"):
        zone = "main"
    pmode = str(raw.get("placement_mode") or ("center" if anchor == "center" else ("corner" if anchor.startswith("corner_") else "wall"))).lower().strip()
    if pmode not in ("wall", "corner", "center", "attached"):
        pmode = "wall"
    attach_slot = raw.get("attach_slot")
    if attach_slot is not None:
        attach_slot = str(attach_slot).lower().strip()
        if attach_slot not in ATTACH_SLOTS:
            attach_slot = None
    parent_ref = raw.get("parent_ref")
    parent_ref = str(parent_ref).strip()[:24] if parent_ref else None
    try:
        slot_order = int(raw.get("wall_slot_order") if raw.get("wall_slot_order") is not None else idx)
    except Exception:
        slot_order = idx

    mount = str(raw.get("mount") or ("wall_high" if ftype == "wall_cabinet" else "floor")).lower().strip()
    if mount not in ("floor", "wall_high"):
        mount = "floor"

    return {
        "id": f"item_{uuid.uuid4().hex[:8]}",
        "type": ftype,
        "label": label,
        "color_hex": str(raw.get("color_hex") or "#D8A36C")[:9],
        "appearance": str(raw.get("appearance") or "").strip()[:160],
        "seen_in_photos": sorted(set(seen)) or [0],
        "best_photo_index": int(clamp(fnum("best_photo_index", 0), 0, 30)),
        "bbox_1000": bb,
        "zone": zone,
        "placement_mode": pmode,
        "wall_anchor": anchor,
        "wall_slot_order": slot_order,
        "along_wall_offset": clamp(fnum("along_wall_offset", 0.0), -1.0, 1.0),
        "center_z_offset": clamp(fnum("center_z_offset", 0.0), -1.0, 1.0),
        "parent_ref": parent_ref,
        "attach_slot": attach_slot,
        "facing": str(raw.get("facing") or "").lower().strip(),
        "facing_target": str(raw.get("facing_target") or "").strip()[:22],
        "mount": mount,
        "frontal_width_m": clamp(fnum("frontal_width_m", dw), 0.2, 6.0),
        "height_m": clamp(fnum("height_m", dh), 0.02, 3.0),
        "depth_m": clamp(fnum("depth_m", dd), 0.08, 3.0),
        "is_glass": bool(raw.get("is_glass", False)),
        "confidence": clamp(fnum("confidence", 0.6), 0, 1),
        "index": idx,
    }


def _clean_wall_features(raw_wf: dict | None) -> dict:
    out = {"windows": [], "subzone_split": None}
    if not isinstance(raw_wf, dict):
        return out
    for w in (raw_wf.get("windows") or [])[:4]:
        if not isinstance(w, dict):
            continue
        wall = str(w.get("wall") or "").lower().strip()
        if wall not in ("north", "south", "west", "east"):
            continue
        try:
            off = clamp(float(w.get("offset", 0.5)), 0.12, 0.88)
        except Exception:
            off = 0.5
        try:
            wm = clamp(float(w.get("width_m", 1.3)), 0.65, 2.6)
        except Exception:
            wm = 1.3
        out["windows"].append({
            "wall": wall,
            "offset": round(off, 2),
            "width_m": round(wm, 2),
            "has_curtain": bool(w.get("has_curtain", True)),
            "evidence": str(w.get("evidence") or "")[:160],
        })
    sz = raw_wf.get("subzone_split")
    if isinstance(sz, dict) and sz.get("has_connected_balcony"):
        end = str(sz.get("balcony_end") or "").lower().strip()
        if end in ("north", "south", "west", "east"):
            try:
                ratio = clamp(float(sz.get("balcony_ratio", 0.27)), 0.20, 0.36)
            except Exception:
                ratio = 0.27
            out["subzone_split"] = {
                "has_connected_balcony": True,
                "balcony_end": end,
                "ratio": round(ratio, 3),
                "divider_kind": "open_arch",
                "has_curtain": bool(sz.get("has_curtain", True)),
                "evidence": str(sz.get("evidence") or "")[:160],
            }
    return out


def analyze_room_pass1(
    proj: dict,
    room: dict,
    photo_abs: list,
    model: str = None,
    orientations: list = None,
    crop_map_abs: str | None = None,
) -> dict:
    if not photo_abs:
        return {"furniture": [], "wall_hex": "#F2EEE7", "floor_hex": "#C9A176", "align_summary": "无照片"}
    walls = describe_walls(room, proj.get("rooms", []))
    has_map = bool(crop_map_abs and os.path.isfile(crop_map_abs))
    prompt = _pass1_prompt(room, walls, len(photo_abs), orientations, has_map_img=has_map)
    imgs = ([crop_map_abs] if has_map else []) + photo_abs[:10]
    spec = call_vlm_json(
        prompt,
        imgs,
        timeout=240,
        max_images=len(imgs),
        model=model or ROOM_VLM_MODEL,
        max_tokens=16384,
    )
    items = [x for x in (_normalize_item(r, i) for i, r in enumerate(spec.get("furniture") or [])) if x]
    for it in items:
        it["best_photo_index"] = int(clamp(it["best_photo_index"], 0, len(photo_abs) - 1))
        it["seen_in_photos"] = [p for p in it["seen_in_photos"] if 0 <= p < len(photo_abs)] or [it["best_photo_index"]]
    spec["furniture"] = items[:14]
    spec["wall_features_detected"] = _clean_wall_features(spec.get("wall_features_detected"))
    return spec


# --------------------------------------------------------------------------- merge


def _same_object(a: dict, b: dict) -> bool:
    if a["type"] != b["type"]:
        return False
    # 若两者出现在同一张实景照片里，且检测框在同张图中明显分离，则绝对是两件独立物体（例如 4 把餐椅 / 2 个床头柜）
    shared_photos = set(a.get("seen_in_photos") or []) & set(b.get("seen_in_photos") or [])
    if shared_photos and a.get("best_photo_index") == b.get("best_photo_index"):
        if a.get("bbox_1000") and b.get("bbox_1000") and _bbox_iou_1000(a["bbox_1000"], b["bbox_1000"]) < 0.25:
            return False
    if a["type"] == "chair" and shared_photos:
        return False
    if a["wall_anchor"] != b["wall_anchor"]:
        return False
    if abs(a["along_wall_offset"] - b["along_wall_offset"]) > 0.26:
        return False
    if _color_dist(a.get("color_hex", ""), b.get("color_hex", "")) > 70:
        return False
    for k in ("frontal_width_m", "depth_m"):
        va, vb = a.get(k) or 0.1, b.get(k) or 0.1
        if max(va, vb) / max(0.1, min(va, vb)) > 1.55:
            return False
    return True


def merge_multiview(items: list) -> list:
    """同一件家具跨照片合并：同类型 + 同贴墙 + 沿墙位置接近 + 颜色接近 + 尺寸接近。"""
    merged = []
    for it in sorted(items, key=lambda x: (-len(x.get("seen_in_photos") or []), -(x.get("confidence") or 0))):
        hit = None
        for m in merged:
            if _same_object(it, m):
                hit = m
                break
        if not hit:
            merged.append(dict(it))
            continue
        hit["seen_in_photos"] = sorted(set(hit["seen_in_photos"]) | set(it["seen_in_photos"]))
        hit["views"] = len(hit["seen_in_photos"])
        if (it.get("bbox_1000") and not hit.get("bbox_1000")) or (
            it.get("confidence", 0) > hit.get("confidence", 0) + 0.08 and it.get("bbox_1000")
        ):
            hit["bbox_1000"] = it["bbox_1000"]
            hit["best_photo_index"] = it["best_photo_index"]
        hit["confidence"] = max(hit.get("confidence", 0), it.get("confidence", 0))
        hit["frontal_width_m"] = round(median([hit["frontal_width_m"], it["frontal_width_m"]], hit["frontal_width_m"]), 2)
        hit["height_m"] = round(median([hit["height_m"], it["height_m"]], hit["height_m"]), 2)
        hit["depth_m"] = round(median([hit["depth_m"], it["depth_m"]], hit["depth_m"]), 2)
        if not hit.get("parent_ref") and it.get("parent_ref"):
            hit["parent_ref"] = it["parent_ref"]
        if not hit.get("attach_slot") and it.get("attach_slot"):
            hit["attach_slot"] = it["attach_slot"]
        if hit.get("zone") == "main" and it.get("zone") == "balcony":
            hit["zone"] = "balcony"
        if len(it.get("appearance") or "") > len(hit.get("appearance") or ""):
            hit["appearance"] = it["appearance"]
    for i, it in enumerate(merged):
        it.setdefault("views", len(it.get("seen_in_photos") or [1]))
        it["_merge_index"] = i
    return merged


# --------------------------------------------------------------------------- pass2


def audit_room_pass2(
    proj: dict,
    room: dict,
    photo_abs: list,
    merged: list,
    model: str = None,
    walls_desc: str = "",
    orientations: list = None,
    crop_map_abs: str | None = None,
    wall_features: dict | None = None,
) -> tuple:
    if not merged or not photo_abs:
        return merged, {"notes": []}
    has_map = bool(crop_map_abs and os.path.isfile(crop_map_abs))
    prompt = _pass2_prompt(merged, walls_desc, orientations, wall_features, len(photo_abs), has_map_img=has_map)
    imgs = ([crop_map_abs] if has_map else []) + photo_abs[:10]
    audit = call_vlm_json(
        prompt,
        imgs,
        timeout=240,
        max_images=len(imgs),
        model=model or AUDIT_VLM_MODEL,
        max_tokens=16384,
    )
    by_id = {it["id"]: it for it in merged}
    dropped = set()
    for c in audit.get("corrections") or []:
        if not isinstance(c, dict):
            continue
        it = by_id.get(str(c.get("id")))
        if not it:
            continue
        if c.get("exists") is False or c.get("duplicate_of"):
            dropped.add(it["id"])
            continue
        if str(c.get("wall_anchor") or "") in ANCHOR_TO_ROT:
            it["wall_anchor"] = str(c["wall_anchor"])
        if str(c.get("zone") or "") in ("main", "balcony"):
            it["zone"] = str(c["zone"])
        if str(c.get("placement_mode") or "") in ("wall", "corner", "center", "attached"):
            it["placement_mode"] = str(c["placement_mode"])
        if c.get("wall_slot_order") is not None:
            try:
                it["wall_slot_order"] = int(c["wall_slot_order"])
            except Exception:
                pass
        if "parent_ref" in c:
            it["parent_ref"] = str(c["parent_ref"]).strip()[:24] if c["parent_ref"] else None
        if "attach_slot" in c:
            aslot = str(c["attach_slot"]).lower().strip() if c["attach_slot"] else None
            it["attach_slot"] = aslot if aslot in ATTACH_SLOTS else None
        if str(c.get("mount") or "") in ("floor", "wall_high"):
            it["mount"] = str(c["mount"])
        for src, dst, lo, hi in (
            ("along_wall_offset", "along_wall_offset", -1, 1),
            ("center_z_offset", "center_z_offset", -1, 1),
            ("frontal_width_m", "frontal_width_m", 0.2, 6.0),
            ("height_m", "height_m", 0.02, 3.0),
            ("depth_m", "depth_m", 0.08, 3.0),
        ):
            if c.get(src) is not None:
                try:
                    it[dst] = clamp(float(c[src]), lo, hi)
                except Exception:
                    pass
        if c.get("best_photo_index") is not None:
            try:
                it["best_photo_index"] = int(clamp(float(c["best_photo_index"]), 0, len(photo_abs) - 1))
            except Exception:
                pass
        if isinstance(c.get("bbox_1000"), list) and len(c["bbox_1000"]) == 4:
            try:
                bb = [round(clamp(float(v), 0, 1000), 1) for v in c["bbox_1000"]]
                if abs(bb[2] - bb[0]) > 10 and abs(bb[3] - bb[1]) > 10:
                    it["bbox_1000"] = bb
            except Exception:
                pass
        if c.get("label"):
            it["label"] = str(c["label"])[:22]
        if c.get("facing") is not None:
            fv = str(c.get("facing") or "").lower()
            if fv in ("room_center", "table", "sofa", "bed", "tv", "none") or fv in FACING_TO_ROT:
                it["facing"] = fv
        if c.get("facing_target") is not None:
            it["facing_target"] = str(c.get("facing_target") or "")[:22]
        it["audit_reason"] = str(c.get("reason") or "")[:120]
    out = [it for it in merged if it["id"] not in dropped]
    for raw in audit.get("missing") or []:
        extra = _normalize_item(raw, len(out))
        if extra:
            extra["best_photo_index"] = int(clamp(extra["best_photo_index"], 0, len(photo_abs) - 1))
            out.append(extra)
    out = merge_multiview(out) if out else []
    return out, audit


# --------------------------------------------------------------------------- 几何求解（主从依附组 + 子区边界）


def build_door_clearance_zones(rw: float, rd: float, openings: list) -> list:
    zones = []
    for op in openings or []:
        kind = op.get("kind")
        if kind not in ("door", "sliding_door", "entry_door"):
            continue
        wall = op.get("wall")
        off = float(op.get("offset", 0.5))
        ow = float(op.get("width_m", 0.9))
        swing = 0.80 if kind in ("door", "entry_door") else 0.36
        if wall == "north":
            zones.append((-rw / 2 + off * rw, -rd / 2 + swing / 2, ow + 0.08, swing))
        elif wall == "south":
            zones.append((-rw / 2 + off * rw, rd / 2 - swing / 2, ow + 0.08, swing))
        elif wall == "west":
            zones.append((-rw / 2 + swing / 2, -rd / 2 + off * rd, swing, ow + 0.08))
        elif wall == "east":
            zones.append((rw / 2 - swing / 2, -rd / 2 + off * rd, swing, ow + 0.08))
    return zones


def _compute_subzone_rects(rw: float, rd: float, subzone_split: dict | None) -> tuple[list, list | None]:
    """返回 (main_rect [x1,z1,x2,z2], balcony_rect | None)。"""
    full = [-rw / 2.0, -rd / 2.0, rw / 2.0, rd / 2.0]
    if not (isinstance(subzone_split, dict) and subzone_split.get("has_connected_balcony")):
        return full, None
    end = subzone_split.get("balcony_end")
    ratio = clamp(float(subzone_split.get("ratio") or 0.27), 0.18, 0.38)
    gap = 0.07  # 隔断垭口帘墙厚度留白
    if end == "south" and rd >= 2.8:
        z_split = rd / 2.0 - rd * ratio
        return [-rw / 2.0, -rd / 2.0, rw / 2.0, z_split - gap], [-rw / 2.0, z_split + gap, rw / 2.0, rd / 2.0]
    if end == "north" and rd >= 2.8:
        z_split = -rd / 2.0 + rd * ratio
        return [-rw / 2.0, z_split + gap, rw / 2.0, rd / 2.0], [-rw / 2.0, -rd / 2.0, rw / 2.0, z_split - gap]
    if end == "east" and rw >= 2.8:
        x_split = rw / 2.0 - rw * ratio
        return [-rw / 2.0, -rd / 2.0, x_split - gap, rd / 2.0], [x_split + gap, -rd / 2.0, rw / 2.0, rd / 2.0]
    if end == "west" and rw >= 2.8:
        x_split = -rw / 2.0 + rw * ratio
        return [x_split + gap, -rd / 2.0, rw / 2.0, rd / 2.0], [-rw / 2.0, -rd / 2.0, x_split - gap, rd / 2.0]
    return full, None


def _resolve_item_zone(it: dict, ftype: str, has_balcony: bool) -> str:
    if not has_balcony:
        return "main"
    label = str(it.get("label") or "")
    if ftype in ("washer", "drying_rack") or any(k in label for k in ("洗衣", "晾衣", "阳台")):
        return "balcony"
    if ftype in ("bed", "wardrobe", "desk", "tv", "tv_stand", "sofa", "dining_table", "coffee_table", "fridge", "nightstand"):
        return "main"
    return "balcony" if it.get("zone") == "balcony" else "main"


def _resolve_parent_links(nodes: list):
    """解析并补全父子依附关系，防止成环。"""
    by_id = {n["_raw"].get("id"): idx for idx, n in enumerate(nodes) if n["_raw"].get("id")}
    by_label = [(idx, norm_label(n["_raw"].get("label") or "")) for idx, n in enumerate(nodes)]

    def find_by_ref(ref: str, child_idx: int) -> int | None:
        if not ref:
            return None
        if ref in by_id and by_id[ref] != child_idx:
            return by_id[ref]
        nref = norm_label(ref)
        for idx, lbl in by_label:
            if idx != child_idx and lbl and (nref == lbl or nref in lbl or lbl in nref):
                return idx
        return None

    def find_by_types(types: tuple, child_idx: int, prefer_anchor: str = None) -> int | None:
        cands = [i for i, n in enumerate(nodes) if i != child_idx and n["type"] in types]
        if not cands:
            return None
        if prefer_anchor:
            same = [i for i in cands if nodes[i]["anchor"] == prefer_anchor]
            if same:
                return same[0]
        return cands[0]

    for idx, n in enumerate(nodes):
        raw = n["_raw"]
        ftype = n["type"]
        label = str(raw.get("label") or "")
        pref = raw.get("parent_ref")
        aslot = raw.get("attach_slot")
        facing = str(raw.get("facing") or "").lower()
        ftarget = str(raw.get("facing_target") or "")

        CHILD_ELIGIBLE = ("chair", "tv", "nightstand", "side_table", "other")
        PRIMARY_ANCHOR_TYPES = (
            "bed", "sofa", "wardrobe", "dining_table", "desk", "fridge", "washer",
            "kitchen_counter", "wall_cabinet", "toilet", "bathtub", "vanity_sink",
            "bookshelf", "cabinet", "tv_stand", "drying_rack",
        )
        if ftype in PRIMARY_ANCHOR_TYPES:
            pidx = None
        else:
            pidx = find_by_ref(pref, idx) if pref else None
            # 仅允许椅子/电视/床头柜通过 facing_target 兜底绑定父物体，绝不把其它主件绑成子件
            if pidx is None and ftarget and ftype in ("chair", "tv", "nightstand"):
                pidx = find_by_ref(ftarget, idx)

        # 语义校验与兜底：餐椅 -> 餐桌；书桌椅 -> 书桌；电视 -> 电视柜；床头柜 -> 床
        if ftype == "chair":
            if pidx is not None and nodes[pidx]["type"] not in ("dining_table", "coffee_table", "desk"):
                pidx = None
            if pidx is None and (
                aslot == "around_table"
                or facing in ("table", "dining_table")
                or "餐" in label
            ):
                pidx = find_by_types(("dining_table",), idx)
                if pidx is not None:
                    aslot = "around_table"
            elif pidx is None and (
                aslot == "front_of_desk"
                or facing == "desk"
                or any(k in label for k in ("书桌椅", "电脑椅", "办公椅", "转椅", "学习椅", "梳妆椅"))
            ):
                pidx = find_by_types(("desk",), idx)
                if pidx is not None:
                    aslot = "front_of_desk"
            elif pidx is not None and not aslot:
                aslot = "around_table" if nodes[pidx]["type"] in ("dining_table", "coffee_table") else "front_of_desk"
        elif ftype == "tv":
            if pidx is not None and nodes[pidx]["type"] != "tv_stand":
                pidx = None
            if pidx is None:
                pidx = find_by_types(("tv_stand",), idx, prefer_anchor=n["anchor"])
            if pidx is not None:
                aslot = aslot if aslot in ("above_stand", "wall_high_above") else (
                    "wall_high_above" if n["mount"] == "wall_high" else "above_stand"
                )
            else:
                # 独立电视没有电视柜时强制壁挂高处，绝不落地挡路
                n["mount"] = "wall_high"
                n["elev"] = max(n["elev"], 1.15)
                n["fd"] = min(n["fd"], 0.14)
        elif ftype == "nightstand":
            if pidx is not None and nodes[pidx]["type"] != "bed":
                pidx = None
            if pidx is None:
                pidx = find_by_types(("bed",), idx, prefer_anchor=n["anchor"])
            if pidx is not None and not aslot:
                aslot = "beside_headboard"
        elif ftype in CHILD_ELIGIBLE:
            if pidx is not None and aslot not in ("beside_sofa", "beside_headboard", "above_stand", "on_top"):
                pidx = None
        else:
            pidx = None

        # 避免成环（父节点不能同时是别人的子节点）
        if pidx is not None and nodes[pidx].get("_parent_idx") is not None:
            pidx = None
        if pidx is not None:
            n["_parent_idx"] = pidx
            n["attach_slot"] = aslot or "around_table"
            n["placement_mode"] = "attached"
        else:
            n["_parent_idx"] = None
            n["attach_slot"] = None


def _in_door_zone(xm: float, zm: float, wx: float, wz: float, door_zones: list) -> bool:
    for dcx, dcz, dwx, dwz in door_zones:
        ox = (wx + dwx) * 0.5 + 0.03 - abs(xm - dcx)
        oz = (wz + dwz) * 0.5 + 0.03 - abs(zm - dcz)
        if ox > 0 and oz > 0:
            return True
    return False


def _mount_children(nodes: list, door_zones: list, main_rect: list, balc_rect: list | None):
    """在主家具已就位的基础上，把子家具按父子关系精准挂靠成刚体组合包。"""
    wall_pad = 0.07

    # 1) 餐椅 around_table：整组计算，如有靠墙过死先微调餐桌与椅组整体平移，再环绕就座并面朝桌心
    tables_with_chairs = {}
    for idx, n in enumerate(nodes):
        if n["_parent_idx"] is not None and n["attach_slot"] == "around_table":
            tables_with_chairs.setdefault(n["_parent_idx"], []).append(n)

    for pidx, chairs in tables_with_chairs.items():
        p = nodes[pidx]
        zrect = balc_rect if (p["zone"] == "balcony" and balc_rect) else main_rect
        c_fw = median([c["fw"] for c in chairs], 0.48)
        c_fd = median([c["fd"] for c in chairs], 0.48)
        pull_d = c_fd * 0.34  # 餐椅半推入桌沿，既贴近又不穿模

        # 先尝试在 zone 内为餐桌腾出足以放餐椅的呼吸边距
        need_margin = c_fd * 0.68 + 0.06
        tx_min = zrect[0] + p["wx"] / 2 + wall_pad + (need_margin if len(chairs) >= 2 else 0.0)
        tx_max = zrect[2] - p["wx"] / 2 - wall_pad - (need_margin if len(chairs) >= 2 else 0.0)
        tz_min = zrect[1] + p["wz"] / 2 + wall_pad + (need_margin if len(chairs) >= 2 else 0.0)
        tz_max = zrect[3] - p["wz"] / 2 - wall_pad - (need_margin if len(chairs) >= 2 else 0.0)
        if tx_min <= tx_max:
            p["xm"] = clamp(p["xm"], tx_min, tx_max)
        if tz_min <= tz_max:
            p["zm"] = clamp(p["zm"], tz_min, tz_max)

        px, pz = p["xm"], p["zm"]
        tw, td = p["wx"], p["wz"]

        # 检测哪些侧边被相邻大件（沙发/衣柜/冰箱/橱柜）紧贴占用（< 0.82m 通道不足放椅）
        blocked_sides = set()
        for other in nodes:
            if other is p or other in chairs or other.get("mount") in ("wall_high", "ceiling"):
                continue
            if other["type"] not in ("sofa", "wardrobe", "fridge", "kitchen_counter", "bed", "cabinet", "bookshelf", "desk"):
                continue
            dx_o = other["xm"] - px
            dz_o = other["zm"] - pz
            gap_x = abs(dx_o) - (tw + other["wx"]) * 0.5
            gap_z = abs(dz_o) - (td + other["wz"]) * 0.5
            # 同轴投影有明显重叠，且垂直距离小于椅深+安全间隙
            if gap_z < 0.05 and 0.0 <= gap_x < c_fd + 0.35:
                blocked_sides.add("west" if dx_o < 0 else "east")
            if gap_x < 0.05 and 0.0 <= gap_z < c_fd + 0.35:
                blocked_sides.add("north" if dz_o < 0 else "south")

        def _chair_side_hint(ch):
            raw_c = ch["_raw"]
            txt = f"{raw_c.get('label') or ''} {raw_c.get('wall_anchor') or ''}".lower()
            if any(k in txt for k in ("东", "east")):
                return "east"
            if any(k in txt for k in ("西", "west")):
                return "west"
            if any(k in txt for k in ("南", "south")):
                return "south"
            if any(k in txt for k in ("北", "north")):
                return "north"
            return None

        # 按方位分组建立高保真槽位：优先尊重 VLM 指明的方向（东/西/南/北），并避开被沙发贴占的一侧
        side_slots = {
            "west": [(px - tw / 2 - pull_d, pz - min(td * 0.27, max(0.24, td * 0.5 - c_fw * 0.58)), "w1"),
                     (px - tw / 2 - pull_d, pz + min(td * 0.27, max(0.24, td * 0.5 - c_fw * 0.58)), "w2")],
            "east": [(px + tw / 2 + pull_d, pz - min(td * 0.27, max(0.24, td * 0.5 - c_fw * 0.58)), "e1"),
                     (px + tw / 2 + pull_d, pz + min(td * 0.27, max(0.24, td * 0.5 - c_fw * 0.58)), "e2")],
            "north": [(px, pz - td / 2 - pull_d, "n1"),
                      (px - min(tw * 0.27, max(0.24, tw * 0.5 - c_fw * 0.58)), pz - td / 2 - pull_d, "n2")],
            "south": [(px, pz + td / 2 + pull_d, "s1"),
                      (px + min(tw * 0.27, max(0.24, tw * 0.5 - c_fw * 0.58)), pz + td / 2 + pull_d, "s2")],
        }
        if tw >= td:
            side_slots["north"] = [
                (px - min(tw * 0.27, max(0.24, tw * 0.5 - c_fw * 0.58)), pz - td / 2 - pull_d, "n1"),
                (px + min(tw * 0.27, max(0.24, tw * 0.5 - c_fw * 0.58)), pz - td / 2 - pull_d, "n2"),
            ]
            side_slots["south"] = [
                (px - min(tw * 0.27, max(0.24, tw * 0.5 - c_fw * 0.58)), pz + td / 2 + pull_d, "s1"),
                (px + min(tw * 0.27, max(0.24, tw * 0.5 - c_fw * 0.58)), pz + td / 2 + pull_d, "s2"),
            ]
            side_slots["west"] = [(px - tw / 2 - pull_d, pz, "w1")]
            side_slots["east"] = [(px + tw / 2 + pull_d, pz, "e1")]

        def slot_ok(cx, cz):
            in_bounds = (
                zrect[0] + c_fw / 2 + 0.03 <= cx <= zrect[2] - c_fw / 2 - 0.03
                and zrect[1] + c_fd / 2 + 0.03 <= cz <= zrect[3] - c_fd / 2 - 0.03
            )
            return in_bounds and not _in_door_zone(cx, cz, c_fw, c_fd, door_zones)

        hinted_count = sum(1 for c in chairs if _chair_side_hint(c) is not None)
        assigned_slots = [None] * len(chairs)
        used_keys = set()

        if hinted_count >= max(2, len(chairs) // 2):
            # 按同侧椅子数量居中重排槽位（若某侧只有1把则居中，若有2把则成对）
            by_side = {}
            for k, ch in enumerate(chairs):
                sd = _chair_side_hint(ch)
                if sd:
                    by_side.setdefault(sd, []).append(k)
            for sd, idxs in by_side.items():
                if sd in ("west", "east"):
                    if len(idxs) == 1:
                        s_pool = [(px + (-tw / 2 - pull_d if sd == "west" else tw / 2 + pull_d), pz, f"{sd}c")]
                    else:
                        off_z2 = min(td * 0.27, max(0.24, td * 0.5 - c_fw * 0.58))
                        x_s = px + (-tw / 2 - pull_d if sd == "west" else tw / 2 + pull_d)
                        s_pool = [(x_s, pz - off_z2, f"{sd}1"), (x_s, pz + off_z2, f"{sd}2")]
                else:
                    if len(idxs) == 1:
                        s_pool = [(px, pz + (-td / 2 - pull_d if sd == "north" else td / 2 + pull_d), f"{sd}c")]
                    else:
                        off_x2 = min(tw * 0.27, max(0.24, tw * 0.5 - c_fw * 0.58))
                        z_s = pz + (-td / 2 - pull_d if sd == "north" else td / 2 + pull_d)
                        s_pool = [(px - off_x2, z_s, f"{sd}1"), (px + off_x2, z_s, f"{sd}2")]
                for m, k in enumerate(idxs):
                    if m < len(s_pool):
                        assigned_slots[k] = s_pool[m]
                        used_keys.add(s_pool[m][2])

        # 其余未指明方位的餐椅从空闲且未被阻挡的槽位补齐
        order_sides = (["north", "south", "west", "east"] if tw >= td else ["east", "west", "north", "south"])
        order_sides = [s for s in order_sides if s not in blocked_sides] + [s for s in order_sides if s in blocked_sides]
        fallback_pool = []
        for s in order_sides:
            for cand in side_slots[s]:
                if cand[2] not in used_keys and slot_ok(cand[0], cand[1]):
                    fallback_pool.append(cand)
        if len(fallback_pool) < len(chairs):
            for s in order_sides:
                for cand in side_slots[s]:
                    if cand[2] not in used_keys and cand not in fallback_pool:
                        fallback_pool.append(cand)

        fb_i = 0
        for k, ch in enumerate(chairs):
            sl = assigned_slots[k]
            if sl is None:
                sl = fallback_pool[fb_i % max(1, len(fallback_pool))] if fallback_pool else (px, pz + td / 2 + pull_d, "f")
                fb_i += 1
            sx, sz, _ = sl
            ch["xm"] = clamp(sx, ch["min_x"], ch["max_x"])
            ch["zm"] = clamp(sz, ch["min_z"], ch["max_z"])
            ch["anchor"] = "center"
            ch["rot_deg"] = int(round(math.degrees(math.atan2(px - ch["xm"], pz - ch["zm"])))) % 360

    # 2) 书桌椅 front_of_desk：摆在书桌正面外侧 0.42m 处并正对书桌
    for n in nodes:
        if n["_parent_idx"] is None or n["attach_slot"] != "front_of_desk":
            continue
        p = nodes[n["_parent_idx"]]
        rad = math.radians(p["rot_deg"])
        fx, fz = math.sin(rad), math.cos(rad)
        dist = p["fd"] * 0.5 + 0.42
        n["xm"] = clamp(p["xm"] + fx * dist, n["min_x"], n["max_x"])
        n["zm"] = clamp(p["zm"] + fz * dist, n["min_z"], n["max_z"])
        n["anchor"] = "center"
        n["rot_deg"] = (p["rot_deg"] + 180) % 360

    # 3) 电视 above_stand / wall_high_above：直接挂载在电视柜正上方（同墙同朝向）
    for n in nodes:
        if n["_parent_idx"] is None or n["attach_slot"] not in ("above_stand", "wall_high_above"):
            continue
        p = nodes[n["_parent_idx"]]
        n["xm"] = p["xm"]
        n["zm"] = p["zm"]
        n["rot_deg"] = p["rot_deg"]
        n["anchor"] = p["anchor"]
        n["wx"] = n["fw"] if n["rot_deg"] in (0, 180) else n["fd"]
        n["wz"] = n["fd"] if n["rot_deg"] in (0, 180) else n["fw"]
        if n["attach_slot"] == "wall_high_above":
            n["mount"] = "wall_high"
            n["elev"] = max(p["fh"] + 0.14, 1.15)
        else:
            n["mount"] = "wall_high"  # 用高程悬置避免落地碰撞
            n["elev"] = round(max(0.45, p["fh"]), 3)

    # 4) 床头柜 beside_headboard：紧贴床头同侧墙身旁
    beds_ns = {}
    for n in nodes:
        if n["_parent_idx"] is not None and str(n["attach_slot"]).startswith("beside_headboard"):
            beds_ns.setdefault(n["_parent_idx"], []).append(n)

    for pidx, nss in beds_ns.items():
        p = nodes[pidx]
        p_anchor = p["anchor"]
        p_rot = p["rot_deg"]
        for k, ns in enumerate(nss):
            ns["anchor"] = p_anchor
            ns["rot_deg"] = p_rot
            ns["wx"] = ns["fw"] if p_rot in (0, 180) else ns["fd"]
            ns["wz"] = ns["fd"] if p_rot in (0, 180) else ns["fw"]
            # 重新按床的墙向计算床头贴墙坐标
            if p_anchor in ("west", "east"):
                ns["xm"] = ns["min_x"] if p_anchor == "west" else ns["max_x"]
                gap_z = p["wz"] * 0.5 + ns["wz"] * 0.5 + 0.05
                prefer_neg = (ns["attach_slot"] == "beside_headboard_left") or (ns["along"] < p["along"]) or (k % 2 == 0 and len(nss) > 1)
                c1 = p["zm"] - gap_z if prefer_neg else p["zm"] + gap_z
                c2 = p["zm"] + gap_z if prefer_neg else p["zm"] - gap_z
                if ns["min_z"] <= c1 <= ns["max_z"] and not _in_door_zone(ns["xm"], c1, ns["wx"], ns["wz"], door_zones):
                    ns["zm"] = c1
                elif ns["min_z"] <= c2 <= ns["max_z"] and not _in_door_zone(ns["xm"], c2, ns["wx"], ns["wz"], door_zones):
                    ns["zm"] = c2
                else:
                    ns["zm"] = clamp(c1, ns["min_z"], ns["max_z"])
            else:
                ns["zm"] = ns["min_z"] if p_anchor in ("north", "corner_nw", "corner_ne") else ns["max_z"]
                gap_x = p["wx"] * 0.5 + ns["wx"] * 0.5 + 0.05
                prefer_neg = (ns["attach_slot"] == "beside_headboard_left") or (ns["along"] < p["along"]) or (k % 2 == 0 and len(nss) > 1)
                c1 = p["xm"] - gap_x if prefer_neg else p["xm"] + gap_x
                c2 = p["xm"] + gap_x if prefer_neg else p["xm"] - gap_x
                if ns["min_x"] <= c1 <= ns["max_x"] and not _in_door_zone(c1, ns["zm"], ns["wx"], ns["wz"], door_zones):
                    ns["xm"] = c1
                elif ns["min_x"] <= c2 <= ns["max_x"] and not _in_door_zone(c2, ns["zm"], ns["wx"], ns["wz"], door_zones):
                    ns["xm"] = c2
                else:
                    ns["xm"] = clamp(c1, ns["min_x"], ns["max_x"])


def solve_room_layout(
    rw: float, rd: float, raw_items: list, openings: list = None, subzone_split: dict = None
) -> list:
    rw, rd = max(1.2, float(rw)), max(1.2, float(rd))
    wall_pad, item_gap = 0.07, 0.08
    door_zones = build_door_clearance_zones(rw, rd, openings or [])
    main_rect, balc_rect = _compute_subzone_rects(rw, rd, subzone_split)
    has_balcony = balc_rect is not None
    solved = []

    for idx, it in enumerate(raw_items):
        ftype = _safe_type(it.get("type"), str(it.get("label") or ""))
        dw, dh, dd = TYPE_DEFAULT_DIMS[ftype]
        fw = float(it.get("frontal_width_m") or dw)
        fh = float(it.get("height_m") or dh)
        fd = float(it.get("depth_m") or dd)

        zone = _resolve_item_zone(it, ftype, has_balcony)
        zrect = balc_rect if (zone == "balcony" and balc_rect) else main_rect
        zw = max(0.8, zrect[2] - zrect[0])
        zd = max(0.8, zrect[3] - zrect[1])

        anchor = it.get("wall_anchor") if it.get("wall_anchor") in ANCHOR_TO_ROT else "north"
        along = clamp(float(it.get("along_wall_offset") or 0.0), -0.94, 0.94)
        mount = str(it.get("mount") or ("wall_high" if ftype == "wall_cabinet" else "floor")).lower()
        label = str(it.get("label") or "")
        if any(k in label for k in ("吊灯", "吸顶灯", "顶灯", "chandelier")):
            mount = "ceiling"
            anchor = "center"
        elif ftype == "wall_cabinet" or any(k in label for k in ("热水器", "空调", "吊柜", "壁挂")):
            mount = "wall_high"
        if ftype == "tv":
            fd = clamp(fd, 0.08, 0.22)
            fw = clamp(fw, 0.65, 1.65)
            fh = clamp(fh, 0.42, 0.95)

        rot_deg = ANCHOR_TO_ROT[anchor]
        raw_facing = str(it.get("facing") or "").lower().strip()
        if anchor in ("corner_nw", "corner_sw") and raw_facing == "east":
            rot_deg = 90
        elif anchor in ("corner_ne", "corner_se") and raw_facing == "west":
            rot_deg = 270
        elif anchor == "center" and ftype == "dining_table" and raw_facing in ("east", "west"):
            rot_deg = FACING_TO_ROT[raw_facing]
        has_long_counter = any(
            _safe_type(x.get("type"), str(x.get("label") or "")) == "kitchen_counter"
            and x.get("wall_anchor") in ("east", "west")
            for x in raw_items
        )
        if (
            ftype == "kitchen_counter"
            and not has_long_counter
            and rd > rw * 1.30
            and fw > rw * 0.85
            and anchor in ("north", "south", "corner_nw", "corner_ne", "corner_sw", "corner_se")
        ):
            anchor, rot_deg = "west", 90
            fw = max(fw, min(rd * 0.72, 3.2))
        if anchor in ("corner_nw", "corner_sw") and rw > rd:
            rot_deg = 90

        # 人体工学尺寸护栏（严禁翻转床的真贴墙方向！仅在合理人体工学区间内微调床长以保留过道）
        if ftype in ("kitchen_counter", "wall_cabinet", "wardrobe", "sofa", "desk", "tv_stand", "bookshelf", "vanity_sink"):
            if fd > fw * 1.35 and fd > 0.9:
                fw, fd = fd, fw
            fd = clamp(fd, 0.30, 0.98 if ftype == "sofa" else 0.66)
        if ftype == "wall_cabinet":
            fd = min(fd, 0.40)
            fh = clamp(fh, 0.45, 0.85)
        elif ftype == "bed":
            fw = clamp(fw, 1.05, 1.88)
            fh = clamp(fh, 0.72, 1.10)
            zone_perp = (zw if rot_deg in (90, 270) else zd) - wall_pad * 2
            fd = clamp(fd, 1.82, 2.08)
            if fd > zone_perp - 0.36:
                fd = max(1.76, zone_perp - 0.36)
        elif ftype == "wardrobe":
            fh = clamp(fh, 1.85, 2.15)
            fd = clamp(fd, 0.50, 0.64)
        elif ftype == "fridge":
            fw = clamp(fw, 0.62, 0.92)
            fh = clamp(fh, 1.60, 1.90)
            fd = clamp(fd, 0.58, 0.74)
        elif ftype == "dining_table":
            fh = clamp(fh, 0.70, 0.80)

        wall_avail = (zw if rot_deg in (0, 180) else zd) - wall_pad * 2 - 0.04
        perp_avail = (zd if rot_deg in (0, 180) else zw) - wall_pad * 2 - 0.08
        fw = min(fw, max(0.28, wall_avail * 0.96))
        fd = min(fd, max(0.08 if ftype == "tv" else 0.24, perp_avail * 0.92))

        wx = fw if rot_deg in (0, 180) else fd
        wz = fd if rot_deg in (0, 180) else fw

        min_x = zrect[0] + wx / 2 + wall_pad
        max_x = zrect[2] - wx / 2 - wall_pad
        min_z = zrect[1] + wz / 2 + wall_pad
        max_z = zrect[3] - wz / 2 - wall_pad
        if min_x > max_x:
            min_x = max_x = (zrect[0] + zrect[2]) * 0.5
        if min_z > max_z:
            min_z = max_z = (zrect[1] + zrect[3]) * 0.5

        half_x = max(0.0, (max_x - min_x) * 0.5)
        half_z = max(0.0, (max_z - min_z) * 0.5)
        mid_x = (min_x + max_x) * 0.5
        mid_z = (min_z + max_z) * 0.5

        if anchor == "north":
            zm, xm = min_z, mid_x + along * half_x
        elif anchor == "south":
            zm, xm = max_z, mid_x + along * half_x
        elif anchor == "west":
            xm, zm = min_x, mid_z + along * half_z
        elif anchor == "east":
            xm, zm = max_x, mid_z + along * half_z
        elif anchor == "corner_nw":
            xm, zm = min_x, min_z
        elif anchor == "corner_ne":
            xm, zm = max_x, min_z
        elif anchor == "corner_sw":
            xm, zm = min_x, max_z
        elif anchor == "corner_se":
            xm, zm = max_x, max_z
        else:
            xm = mid_x + along * half_x * 0.85
            zm = mid_z + float(it.get("center_z_offset") or 0.0) * half_z * 0.85

        if mount == "ceiling":
            elev = float(it.get("elevation_m") or 2.25)
            xm, zm = (zrect[0] + zrect[2]) * 0.5, (zrect[1] + zrect[3]) * 0.5
        elif mount == "wall_high":
            default_elev = 1.90 if any(k in label for k in ("热水器", "空调")) else (1.15 if ftype == "tv" else 1.45)
            elev = float(it.get("elevation_m") or default_elev)
            elev = min(elev, max(0.35, round(2.16 - fh, 3)))
        else:
            elev = float(it.get("elevation_m") or 0.0)

        try:
            slot_order = int(it.get("wall_slot_order") if it.get("wall_slot_order") is not None else idx)
        except Exception:
            slot_order = idx

        solved.append({
            "_raw": it,
            "type": ftype,
            "zone": zone,
            "placement_mode": str(it.get("placement_mode") or ("center" if anchor == "center" else "wall")),
            "mount": mount,
            "anchor": anchor,
            "slot_order": slot_order,
            "along": along,
            "rot_deg": rot_deg,
            "fw": fw,
            "fh": fh,
            "fd": fd,
            "wx": wx,
            "wz": wz,
            "xm": xm,
            "zm": zm,
            "elev": elev,
            "min_x": min_x,
            "max_x": max_x,
            "min_z": min_z,
            "max_z": max_z,
        })

    # 第一步：解析父子依附关系
    _resolve_parent_links(solved)

    # 第二步：仅为带围桌餐椅的主餐桌临时留出椅位呼吸包络，然后先求解全部主家具
    primaries = [n for n in solved if n["_parent_idx"] is None]
    chair_parents = {
        n["_parent_idx"] for n in solved if n["_parent_idx"] is not None and n["attach_slot"] == "around_table"
    }
    orig_dims = {}
    for pidx in chair_parents:
        p = solved[pidx]
        orig_dims[pidx] = (p["wx"], p["wz"], p["min_x"], p["max_x"], p["min_z"], p["max_z"])
        zrect = balc_rect if (p["zone"] == "balcony" and balc_rect) else main_rect
        zw, zd = zrect[2] - zrect[0], zrect[3] - zrect[1]
        p["wx"] = min(zw - 0.20, p["wx"] + 0.56)
        p["wz"] = min(zd - 0.20, p["wz"] + 0.56)
        p["min_x"] = zrect[0] + p["wx"] / 2 + wall_pad
        p["max_x"] = zrect[2] - p["wx"] / 2 - wall_pad
        p["min_z"] = zrect[1] + p["wz"] / 2 + wall_pad
        p["max_z"] = zrect[3] - p["wz"] / 2 - wall_pad
        if p["min_x"] <= p["max_x"]:
            p["xm"] = clamp(p["xm"], p["min_x"], p["max_x"])
        if p["min_z"] <= p["max_z"]:
            p["zm"] = clamp(p["zm"], p["min_z"], p["max_z"])

    for layer in ("floor", "wall_high"):
        group = [n for n in primaries if n["mount"] == layer and n["type"] != "rug"]
        for _ in range(48):
            moved = False
            if layer == "floor" and door_zones:
                for a in group:
                    for dcx, dcz, dwx, dwz in door_zones:
                        ox = (a["wx"] + dwx) * 0.5 + 0.04 - abs(a["xm"] - dcx)
                        oz = (a["wz"] + dwz) * 0.5 + 0.04 - abs(a["zm"] - dcz)
                        if ox > 0 and oz > 0:
                            if a["anchor"] in ("north", "south", "corner_nw", "corner_ne", "corner_sw", "corner_se"):
                                a["xm"] += (1 if a["xm"] >= dcx else -1) * ox
                            elif a["anchor"] in ("west", "east"):
                                a["zm"] += (1 if a["zm"] >= dcz else -1) * oz
                            elif ox < oz:
                                a["xm"] += (1 if a["xm"] >= dcx else -1) * ox
                            else:
                                a["zm"] += (1 if a["zm"] >= dcz else -1) * oz
                            a["xm"] = clamp(a["xm"], a["min_x"], a["max_x"])
                            a["zm"] = clamp(a["zm"], a["min_z"], a["max_z"])
                            moved = True
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    ox = (a["wx"] + b["wx"]) * 0.5 + item_gap - abs(a["xm"] - b["xm"])
                    oz = (a["wz"] + b["wz"]) * 0.5 + item_gap - abs(a["zm"] - b["zm"])
                    if ox <= 0 or oz <= 0:
                        continue
                    same_ew = (a["anchor"] == b["anchor"]) and a["anchor"] in ("west", "east") and a["zone"] == b["zone"]
                    same_ns = (a["anchor"] == b["anchor"]) and a["anchor"] in ("north", "south") and a["zone"] == b["zone"]
                    if same_ew:
                        first, second = (a, b) if (a["slot_order"], a["along"]) <= (b["slot_order"], b["along"]) else (b, a)
                        req = (first["wz"] + second["wz"]) * 0.5 + item_gap
                        mid = (first["zm"] + second["zm"]) * 0.5
                        first["zm"], second["zm"] = mid - req * 0.5, mid + req * 0.5
                    elif same_ns:
                        first, second = (a, b) if (a["slot_order"], a["along"]) <= (b["slot_order"], b["along"]) else (b, a)
                        req = (first["wx"] + second["wx"]) * 0.5 + item_gap
                        mid = (first["xm"] + second["xm"]) * 0.5
                        first["xm"], second["xm"] = mid - req * 0.5, mid + req * 0.5
                    elif a["anchor"] in ("north", "south") and b["anchor"] in ("east", "west"):
                        # 垂直墙 L 形角碰撞：各自沿自身墙向滑离转角
                        a["xm"] += (-1.0 if b["anchor"] == "east" else 1.0) * ox * 0.65
                        b["zm"] += (-1.0 if a["anchor"] == "south" else 1.0) * oz * 0.65
                    elif b["anchor"] in ("north", "south") and a["anchor"] in ("east", "west"):
                        b["xm"] += (-1.0 if a["anchor"] == "east" else 1.0) * ox * 0.65
                        a["zm"] += (-1.0 if b["anchor"] == "south" else 1.0) * oz * 0.65
                    elif ox <= oz:
                        sgn = -1.0 if a["xm"] <= b["xm"] else 1.0
                        a["xm"] += sgn * ox * 0.52
                        b["xm"] -= sgn * ox * 0.52
                    else:
                        sgn = -1.0 if a["zm"] <= b["zm"] else 1.0
                        a["zm"] += sgn * oz * 0.52
                        b["zm"] -= sgn * oz * 0.52
                    for node in (a, b):
                        node["xm"] = clamp(node["xm"], node["min_x"], node["max_x"])
                        node["zm"] = clamp(node["zm"], node["min_z"], node["max_z"])
                        if node["anchor"] == "north":
                            node["zm"] = node["min_z"]
                        elif node["anchor"] == "south":
                            node["zm"] = node["max_z"]
                        elif node["anchor"] == "west":
                            node["xm"] = node["min_x"]
                        elif node["anchor"] == "east":
                            node["xm"] = node["max_x"]
                    # 若墙贴边件仍因过长（如通长橱柜/吊柜）卡死在边界且相互穿插，裁短较长件的沿墙长度让出转角/油烟机空间
                    ox2 = (a["wx"] + b["wx"]) * 0.5 + item_gap - abs(a["xm"] - b["xm"])
                    oz2 = (a["wz"] + b["wz"]) * 0.5 + item_gap - abs(a["zm"] - b["zm"])
                    if ox2 > 0.02 and oz2 > 0.02:
                        # 判断是否为贴边柜体/长台面
                        if same_ew or same_ns:
                            longer, shorter = (a, b) if a["fw"] >= b["fw"] else (b, a)
                            if longer["fw"] > 0.90:
                                shrink = min(longer["fw"] - 0.65, (oz2 if same_ew else ox2))
                                if shrink > 0.02:
                                    longer["fw"] -= shrink
                                    if same_ew:
                                        longer["wz"] = longer["fw"]
                                        zrect = balc_rect if (longer["zone"] == "balcony" and balc_rect) else main_rect
                                        longer["min_z"] = zrect[1] + longer["wz"] / 2 + wall_pad
                                        longer["max_z"] = zrect[3] - longer["wz"] / 2 - wall_pad
                                        longer["zm"] += (-1.0 if shorter["zm"] >= longer["zm"] else 1.0) * shrink * 0.5
                                        longer["zm"] = clamp(longer["zm"], longer["min_z"], longer["max_z"])
                                    else:
                                        longer["wx"] = longer["fw"]
                                        zrect = balc_rect if (longer["zone"] == "balcony" and balc_rect) else main_rect
                                        longer["min_x"] = zrect[0] + longer["wx"] / 2 + wall_pad
                                        longer["max_x"] = zrect[2] - longer["wx"] / 2 - wall_pad
                                        longer["xm"] += (-1.0 if shorter["xm"] >= longer["xm"] else 1.0) * shrink * 0.5
                                        longer["xm"] = clamp(longer["xm"], longer["min_x"], longer["max_x"])
                        elif (a["anchor"] in ("north", "south") and b["anchor"] in ("east", "west")) or (
                            b["anchor"] in ("north", "south") and a["anchor"] in ("east", "west")
                        ):
                            longer, shorter = (a, b) if a["fw"] >= b["fw"] else (b, a)
                            if longer["fw"] > 1.15 and longer["type"] in ("kitchen_counter", "wall_cabinet", "cabinet", "wardrobe", "bookshelf"):
                                zrect = balc_rect if (longer["zone"] == "balcony" and balc_rect) else main_rect
                                if longer["anchor"] in ("east", "west"):
                                    shrink = min(longer["fw"] - 0.80, oz2)
                                    if shrink > 0.02:
                                        longer["fw"] -= shrink
                                        longer["wz"] = longer["fw"]
                                        longer["min_z"] = zrect[1] + longer["wz"] / 2 + wall_pad
                                        longer["max_z"] = zrect[3] - longer["wz"] / 2 - wall_pad
                                        longer["zm"] += (-1.0 if shorter["anchor"] == "south" else 1.0) * shrink * 0.5
                                        longer["zm"] = clamp(longer["zm"], longer["min_z"], longer["max_z"])
                                else:
                                    shrink = min(longer["fw"] - 0.80, ox2)
                                    if shrink > 0.02:
                                        longer["fw"] -= shrink
                                        longer["wx"] = longer["fw"]
                                        longer["min_x"] = zrect[0] + longer["wx"] / 2 + wall_pad
                                        longer["max_x"] = zrect[2] - longer["wx"] / 2 - wall_pad
                                        longer["xm"] += (-1.0 if shorter["anchor"] == "east" else 1.0) * shrink * 0.5
                                        longer["xm"] = clamp(longer["xm"], longer["min_x"], longer["max_x"])
                    moved = True
            if not moved:
                break

    # 第二步半：开门类家具（衣柜/冰箱/洗衣机）正面 0.42m 门扇开启净空软推让
    floor_primaries = [n for n in primaries if n["mount"] == "floor" and n["type"] != "rug"]
    for op in floor_primaries:
        if op["type"] not in ("wardrobe", "fridge", "washer"):
            continue
        clr = 0.42
        rad = math.radians(op["rot_deg"])
        fx, fz = round(math.sin(rad)), round(math.cos(rad))
        # 计算 op 正前方净空盒中心与尺寸
        if abs(fz) >= abs(fx):
            fcx = op["xm"]
            fcz = op["zm"] + fz * (op["wz"] * 0.5 + clr * 0.5)
            fwx, fwz = max(0.30, op["wx"] - 0.12), clr
        else:
            fcx = op["xm"] + fx * (op["wx"] * 0.5 + clr * 0.5)
            fcz = op["zm"]
            fwx, fwz = clr, max(0.30, op["wz"] - 0.12)
        for b in floor_primaries:
            if b is op or b["zone"] != op["zone"] or b.get("anchor") == op.get("anchor"):
                continue
            ox = (b["wx"] + fwx) * 0.5 - abs(b["xm"] - fcx)
            oz = (b["wz"] + fwz) * 0.5 - abs(b["zm"] - fcz)
            if ox > 0.02 and oz > 0.02:
                # 若 b 贴东/西墙而 op 在北/南墙，则沿 Z 轴把 b（及同墙顺延件）推离 op
                if b["anchor"] in ("west", "east") and abs(fz) > 0:
                    shift_z = (1.0 if fz > 0 else -1.0) * oz
                    same_wall = sorted(
                        [c for c in floor_primaries if c["zone"] == b["zone"] and c["anchor"] == b["anchor"]],
                        key=lambda c: c["zm"],
                        reverse=(shift_z > 0),
                    )
                    # 检查该侧顺延空间是否够用
                    rem = shift_z
                    for c in same_wall:
                        if (shift_z > 0 and c["zm"] < b["zm"] - 1e-3) or (shift_z < 0 and c["zm"] > b["zm"] + 1e-3):
                            continue
                        prev_z = c["zm"]
                        c["zm"] = clamp(c["zm"] + rem, c["min_z"], c["max_z"])
                        # 确保不推入门洞区
                        if any((abs(c["xm"] - dcx) < (c["wx"] + dwx) * 0.5 + 0.02) and (abs(c["zm"] - dcz) < (c["wz"] + dwz) * 0.5 + 0.02) for dcx, dcz, dwx, dwz in door_zones):
                            c["zm"] = prev_z
                            break
                        rem = c["zm"] - prev_z
                        if abs(rem) < 0.01:
                            break
                elif b["anchor"] in ("north", "south") and abs(fx) > 0:
                    shift_x = (1.0 if fx > 0 else -1.0) * ox
                    same_wall = sorted(
                        [c for c in floor_primaries if c["zone"] == b["zone"] and c["anchor"] == b["anchor"]],
                        key=lambda c: c["xm"],
                        reverse=(shift_x > 0),
                    )
                    rem = shift_x
                    for c in same_wall:
                        if (shift_x > 0 and c["xm"] < b["xm"] - 1e-3) or (shift_x < 0 and c["xm"] > b["xm"] + 1e-3):
                            continue
                        prev_x = c["xm"]
                        c["xm"] = clamp(c["xm"] + rem, c["min_x"], c["max_x"])
                        if any((abs(c["xm"] - dcx) < (c["wx"] + dwx) * 0.5 + 0.02) and (abs(c["zm"] - dcz) < (c["wz"] + dwz) * 0.5 + 0.02) for dcx, dcz, dwx, dwz in door_zones):
                            c["xm"] = prev_x
                            break
                        rem = c["xm"] - prev_x
                        if abs(rem) < 0.01:
                            break
        # 推让后保证同墙各件间仍留有 >= item_gap 的间距（紧缩压缩顺延）
        for wall_side in ("west", "east"):
            seq = sorted([c for c in floor_primaries if c["zone"] == op["zone"] and c["anchor"] == wall_side], key=lambda c: c["zm"])
            for k in range(len(seq) - 2, -1, -1):
                cur, nxt = seq[k], seq[k + 1]
                max_allow = nxt["zm"] - (cur["wz"] + nxt["wz"]) * 0.5 - item_gap
                if cur["zm"] > max_allow:
                    cur["zm"] = max(cur["min_z"], max_allow)
            for k in range(len(seq) - 1):
                cur, nxt = seq[k], seq[k + 1]
                min_allow = cur["zm"] + (cur["wz"] + nxt["wz"]) * 0.5 + item_gap
                if nxt["zm"] < min_allow:
                    nxt["zm"] = min(nxt["max_z"], min_allow)
        for wall_side in ("north", "south"):
            seq = sorted([c for c in floor_primaries if c["zone"] == op["zone"] and c["anchor"] == wall_side], key=lambda c: c["xm"])
            for k in range(len(seq) - 2, -1, -1):
                cur, nxt = seq[k], seq[k + 1]
                max_allow = nxt["xm"] - (cur["wx"] + nxt["wx"]) * 0.5 - item_gap
                if cur["xm"] > max_allow:
                    cur["xm"] = max(cur["min_x"], max_allow)
            for k in range(len(seq) - 1):
                cur, nxt = seq[k], seq[k + 1]
                min_allow = cur["xm"] + (cur["wx"] + nxt["wx"]) * 0.5 + item_gap
                if nxt["xm"] < min_allow:
                    nxt["xm"] = min(nxt["max_x"], min_allow)

    for pidx, (wx0, wz0, mnx, mxx, mnz, mxz) in orig_dims.items():
        p = solved[pidx]
        p["wx"], p["wz"] = wx0, wz0
        p["min_x"], p["max_x"], p["min_z"], p["max_z"] = mnx, mxx, mnz, mxz

    # 第三步：将子家具围绕已就位的主家具精确挂载
    _mount_children(solved, door_zones, main_rect, balc_rect)

    # 第四步：最终轴对齐与朝向校验（依附子家具保留精准指向父家具的朝向）
    def _deg(dx, dz):
        return int(round(math.degrees(math.atan2(dx, dz)))) % 360

    axis_types = {
        "bed", "wardrobe", "desk", "sofa", "tv_stand", "tv", "fridge", "kitchen_counter",
        "wall_cabinet", "cabinet", "vanity_sink", "toilet", "bathtub", "washer",
        "drying_rack", "bookshelf", "nightstand", "coffee_table", "side_table", "dining_table", "rug",
    }
    by_label = {norm_label(n["_raw"].get("label") or ""): n for n in solved}
    for n in solved:
        if n["_parent_idx"] is not None:
            continue
        raw = n["_raw"]
        ftype = n["type"]
        facing = str(raw.get("facing") or "").lower().strip()
        rot = None
        if ftype in ("coffee_table", "rug") and n["anchor"] == "center":
            n["rot_deg"] = 0
            continue
        if ftype == "dining_table" and n["anchor"] == "center":
            if facing in ("east", "west"):
                n["rot_deg"] = FACING_TO_ROT[facing]
            else:
                n["rot_deg"] = 0
            continue
        if n["anchor"] != "center" and (ftype in axis_types or facing == "room_center"):
            rot = None
        elif facing in FACING_TO_ROT:
            rot = FACING_TO_ROT[facing]
        elif facing == "room_center":
            rot = _deg(-n["xm"], -n["zm"])
        elif facing in ("table", "dining_table", "coffee_table", "sofa", "bed", "tv", "tv_stand"):
            target = None
            want = str(raw.get("facing_target") or "").strip()
            if want:
                for k, cand in by_label.items():
                    if want and (want in k or k in want):
                        target = cand
                        break
            if not target:
                tmap = {
                    "table": ("dining_table", "coffee_table"),
                    "dining_table": ("dining_table",),
                    "coffee_table": ("coffee_table",),
                    "sofa": ("sofa",),
                    "bed": ("bed",),
                    "tv": ("tv_stand",),
                    "tv_stand": ("tv_stand",),
                }
                target = next((c for c in solved if c["type"] in tmap.get(facing, ())), None)
            if target and target is not n:
                rot = _deg(target["xm"] - n["xm"], target["zm"] - n["zm"])
        if rot is not None:
            n["rot_deg"] = rot
            n["facing_used"] = facing
    return solved


# --------------------------------------------------------------------------- 主流程


def pick_photos(room: dict, pdir: str, limit: int = 10) -> list:
    out = []
    for ph in (room.get("photos") or [])[:limit]:
        p = os.path.join(pdir, ph.get("rel_path", ""))
        if os.path.isfile(p):
            out.append(p)
    return out


def _wall_exposed_intervals(room: dict, all_rooms: list, wall: str) -> list:
    """计算某房间某面墙未被邻接房间遮挡的露外区间 [(start_01, end_01), ...]，用于阶梯户型露外角外窗反哺。"""
    rw, rd = float(room.get("width_m", 3.4)), float(room.get("depth_m", 3.4))
    rx, rz = float(room.get("rel_x", 0.0)), float(room.get("rel_z", 0.0))
    x0, x1 = rx - rw / 2, rx + rw / 2
    z0, z1 = rz - rd / 2, rz + rd / 2
    span = rw if wall in ("north", "south") else rd
    w_a, w_b = (x0, x1) if wall in ("north", "south") else (z0, z1)
    covered = []
    tol = 0.45
    for other in all_rooms:
        if other.get("id") == room.get("id"):
            continue
        ow, od = float(other.get("width_m", 3.4)), float(other.get("depth_m", 3.4))
        ox, oz = float(other.get("rel_x", 0.0)), float(other.get("rel_z", 0.0))
        ox0, ox1 = ox - ow / 2, ox + ow / 2
        oz0, oz1 = oz - od / 2, oz + od / 2
        touch = False
        ca, cb = 0.0, 0.0
        if wall == "north" and abs(oz1 - z0) <= tol:
            touch, ca, cb = True, max(w_a, ox0), min(w_b, ox1)
        elif wall == "south" and abs(oz0 - z1) <= tol:
            touch, ca, cb = True, max(w_a, ox0), min(w_b, ox1)
        elif wall == "west" and abs(ox1 - x0) <= tol:
            touch, ca, cb = True, max(w_a, oz0), min(w_b, oz1)
        elif wall == "east" and abs(ox0 - x1) <= tol:
            touch, ca, cb = True, max(w_a, oz0), min(w_b, oz1)
        if touch and cb - ca > 0.15:
            covered.append((ca, cb))
    covered.sort()
    free = []
    cur = w_a
    for ca, cb in covered:
        if ca > cur + 0.05:
            free.append(((cur - w_a) / span, (ca - w_a) / span))
        cur = max(cur, cb)
    if cur + 0.05 < w_b:
        free.append(((cur - w_a) / span, (w_b - w_a) / span))
    return [(a, b) for a, b in free if (b - a) * span >= 0.72]


def _apply_wall_features(proj: dict, room: dict, wf1: dict | None, wf2: dict | None) -> dict | None:
    """将实景识别到的外窗与连通阳台子区反哺回 room['openings'] 与 room['subzone_split']。"""
    c1 = _clean_wall_features(wf1)
    c2 = _clean_wall_features(wf2)

    all_wins = c2["windows"] if c2["windows"] else c1["windows"]
    all_rooms = proj.get("rooms", [])
    rw, rd = float(room.get("width_m", 3.4)), float(room.get("depth_m", 3.4))
    openings = list(room.get("openings") or [])
    backfilled_wins = []
    win_dir_map = {"north": "N", "south": "S", "west": "W", "east": "E"}

    for w in all_wins:
        wall = w["wall"]
        exposed = _wall_exposed_intervals(room, all_rooms, wall)
        if not exposed:
            continue  # 整面墙紧邻其它房间，绝不补外窗
        if any(op.get("wall") == wall and op.get("kind") == "window" for op in openings):
            continue
        span = rw if wall in ("north", "south") else rd
        off = float(w["offset"])
        # 若该墙部分与邻室相接、部分露外（如阶梯凸角），将窗位夹入最匹配的露外段内
        best_seg = min(exposed, key=lambda s: abs((s[0] + s[1]) * 0.5 - off))
        seg_w = (best_seg[1] - best_seg[0]) * span
        ww = min(float(w["width_m"]), max(0.65, seg_w - 0.14))
        half_01 = (ww * 0.5 + 0.06) / span
        if best_seg[0] + half_01 <= best_seg[1] - half_01:
            off = clamp(off, best_seg[0] + half_01, best_seg[1] - half_01)
        else:
            off = (best_seg[0] + best_seg[1]) * 0.5
        openings.append({
            "kind": "window",
            "wall": wall,
            "offset": round(off, 3),
            "width_m": round(ww, 2),
            "target": "户外",
            "render_leaf": True,
            "backfilled": True,
        })
        backfilled_wins.append(wall)
        if str(room.get("window_dir") or "NONE").upper() == "NONE":
            room["window_dir"] = win_dir_map[wall]

    room["openings"] = openings
    if backfilled_wins:
        log_stage(proj, f"room.window_backfill:{room['name']}", "ok", f"从实景照反哺外窗：{backfilled_wins}")

    subzone = c2["subzone_split"] or c1["subzone_split"]
    rw, rd = float(room.get("width_m", 3.4)), float(room.get("depth_m", 3.4))
    if subzone and subzone.get("has_connected_balcony"):
        end = subzone["balcony_end"]
        axis_len = rd if end in ("north", "south") else rw
        if axis_len >= 3.20:
            room["subzone_split"] = subzone
            log_stage(proj, f"room.subzone:{room['name']}", "ok",
                      f"识别连通阳台子区：{end} 端占比 {subzone['ratio']}（垭口+束帘）")
            return subzone
    room["subzone_split"] = None
    return None


def recompute_world_box(size_m: list, rot_deg: float) -> list:
    w, h, d = [float(x) for x in (size_m or [0.8, 0.8, 0.6])]
    r = int(round(rot_deg)) % 360
    if r in (0, 180):
        return [round(w, 2), round(h, 2), round(d, 2)]
    if r in (90, 270):
        return [round(d, 2), round(h, 2), round(w, 2)]
    rad = math.radians(rot_deg)
    c, s = abs(math.cos(rad)), abs(math.sin(rad))
    return [round(w * c + d * s, 2), round(h, 2), round(w * s + d * c, 2)]


def re_solve_existing_room(room: dict) -> list:
    """对已有 scene_spec.furniture 保留资产/GLB 字段并重跑通用多锚点求解器。"""
    spec = room.get("scene_spec") or {}
    items = list(spec.get("furniture") or [])
    if not items:
        return items
    rw, rd = float(room.get("width_m", 3.4)), float(room.get("depth_m", 3.4))
    subzone_split = room.get("subzone_split") or spec.get("subzone_split")
    id_to_label = {it.get("id"): it.get("label") for it in items if it.get("id")}
    raw_items = []
    for it in items:
        sz = list(it.get("size_m") or [0.8, 0.8, 0.6])
        al = it.get("glb_aligned_size")
        # 若实测 GLB 明显为立式柱体(H > 1.4*W)而 VLM 误估成卧式(W > 1.4*H)，自动按 GLB 真长宽比修正目标框
        if isinstance(al, (list, tuple)) and len(al) == 3 and min(al) > 0.05:
            aw, ah, ad = [float(v) for v in al]
            if (ah > aw * 1.4 and sz[0] > sz[1] * 1.4) or (aw > ah * 1.4 and sz[1] > sz[0] * 1.4):
                scale = max(sz) / max(aw, ah, ad)
                sz = [round(clamp(aw * scale, 0.25, 2.2), 2), round(clamp(ah * scale, 0.25, 2.2), 2), round(clamp(ad * scale, 0.25, 2.2), 2)]
        raw_items.append({
            "type": it.get("type"),
            "label": it.get("label"),
            "color_hex": it.get("color_hex"),
            "appearance": it.get("appearance"),
            "zone": it.get("zone"),
            "placement_mode": it.get("placement_mode"),
            "mount": it.get("mount"),
            "wall_anchor": it.get("wall_anchor"),
            "wall_slot_order": it.get("wall_slot_order"),
            "along_wall_offset": it.get("along_wall_offset"),
            "center_z_offset": it.get("center_z_offset"),
            "parent_ref": id_to_label.get(it.get("parent_id")) or it.get("parent_ref"),
            "attach_slot": it.get("attach_slot"),
            "size_m": sz,
            "elevation_m": it.get("elevation_m"),
            "facing": it.get("facing"),
            "facing_target": it.get("facing_target"),
            "is_glass": it.get("is_glass"),
            "confidence": it.get("confidence"),
        })
    solved = solve_room_layout(rw, rd, raw_items, room.get("openings") or [], subzone_split=subzone_split)
    for it, node in zip(items, solved):
        size_m = [round(node["fw"], 2), round(node["fh"], 2), round(node["fd"], 2)]
        it["zone"] = node["zone"]
        it["placement_mode"] = node["placement_mode"]
        it["mount"] = node["mount"]
        it["wall_anchor"] = node["anchor"]
        it["xm"] = round(node["xm"], 3)
        it["zm"] = round(node["zm"], 3)
        it["elevation_m"] = round(node["elev"], 3)
        it["pos_x"] = round(node["xm"] / rw, 4)
        it["pos_z"] = round(node["zm"] / rd, 4)
        it["size_m"] = size_m
        it["world_box_m"] = [round(node["wx"], 2), round(node["fh"], 2), round(node["wz"], 2)]
        it["rot_deg"] = int(node["rot_deg"]) % 360
    spec["furniture"] = items
    return items


def build_room_spec(proj: dict, room: dict, pdir: str, do_audit: bool = True) -> dict:
    """单房间完整解析：局部标注户型图 → 机位定标 → pass1 → merge → pass2 → 墙面反哺 → 主从求解 → 裁图。"""
    photo_abs = pick_photos(room, pdir)
    if not photo_abs:
        return {"wall_hex": "#F2EEE7", "floor_hex": "#C9A176", "align_summary": "无照片，跳过", "furniture": []}

    fp_abs = None
    fp_url = (proj.get("floorplan") or {}).get("url") or ""
    if fp_url:
        cand = os.path.join(pdir, fp_url.split("/")[-1])
        if os.path.isfile(cand):
            fp_abs = cand

    crop_map_abs = None
    if fp_abs:
        crop_map_abs = render_annotated_room_crop(
            fp_abs,
            room,
            proj.get("rooms", []),
            os.path.join(pdir, "debug", f"room_map_{room['id']}.jpg"),
        )

    orientations = calibrate_photo_orientations(proj, room, fp_abs, photo_abs, crop_map_abs=crop_map_abs) if fp_abs else []
    if orientations:
        log_stage(
            proj,
            f"room.orient:{room['name']}",
            "ok",
            f"{len(orientations)} 张照片完成门窗锚点机位定标",
            photos=[
                {"i": o["photo_index"], "front": o["front_wall"], "left": o["left_wall"], "right": o["right_wall"]}
                for o in orientations
            ][:8],
        )

    spec = analyze_room_pass1(proj, room, photo_abs, orientations=orientations, crop_map_abs=crop_map_abs)
    merged = merge_multiview(spec.get("furniture") or [])
    log_stage(
        proj,
        f"room.pass1:{room['name']}",
        "ok",
        f"{len(photo_abs)} 张照片识别 {len(spec.get('furniture') or [])} 条，跨视角合并后 {len(merged)} 件",
        model=(spec.get("_vlm") or {}).get("model"),
    )

    audit = {}
    if do_audit and merged:
        pass1_merged = list(merged)
        try:
            walls = describe_walls(room, proj.get("rooms", []))
            merged, audit = audit_room_pass2(
                proj,
                room,
                photo_abs,
                merged,
                walls_desc=walls,
                orientations=orientations,
                crop_map_abs=crop_map_abs,
                wall_features=spec.get("wall_features_detected"),
            )
            if not merged and pass1_merged:
                merged = pass1_merged
                pipeline_warning(proj, f"room.audit:{room['name']}", "二审把全部家具都判掉了（疑似过严），已回退到一审合并结果")
            log_stage(
                proj,
                f"room.audit:{room['name']}",
                "ok",
                f"复核后 {len(merged)} 件",
                notes=(audit.get("notes") or [])[:6],
                model=(audit.get("_vlm") or {}).get("model"),
            )
        except Exception as e:  # noqa: BLE001
            pipeline_warning(proj, f"room.audit:{room['name']}", f"复核失败，沿用合并结果：{e}")

    subzone_split = _apply_wall_features(
        proj, room, spec.get("wall_features_detected"), audit.get("wall_features_detected")
    )

    rw, rd = float(room.get("width_m", 3.4)), float(room.get("depth_m", 3.4))
    solved = solve_room_layout(rw, rd, merged, room.get("openings") or [], subzone_split=subzone_split)

    # 先为每个求解节点生成最终 item_id，以便把 parent_id 精确写回 JSON
    node_ids = [f"item_{uuid.uuid4().hex[:8]}" for _ in solved]
    items = []
    for idx, node in enumerate(solved):
        raw = node["_raw"]
        pidx = int(raw.get("best_photo_index") or 0)
        pidx = pidx if 0 <= pidx < len(photo_abs) else 0
        item_id = node_ids[idx]
        crop_rel = f"crops/{item_id}.jpg"
        try:
            crop_furniture_image(photo_abs[pidx], raw.get("bbox_1000"), os.path.join(pdir, crop_rel))
        except Exception:
            crop_rel = None

        size_m = [round(node["fw"], 2), round(node["fh"], 2), round(node["fd"], 2)]
        ctx_rel = f"crops/{item_id}_ctx.jpg"
        try:
            crop_context_image(photo_abs[pidx], raw.get("bbox_1000"), os.path.join(pdir, ctx_rel))
        except Exception:
            ctx_rel = crop_rel
        p_idx = node.get("_parent_idx")
        items.append({
            "id": item_id,
            "type": node["type"],
            "label": str(raw.get("label") or "家具")[:22],
            "color_hex": raw.get("color_hex", "#D8A36C"),
            "appearance": raw.get("appearance", ""),
            "zone": node["zone"],
            "placement_mode": node["placement_mode"],
            "mount": node["mount"],
            "wall_anchor": node["anchor"],
            "wall_slot_order": node["slot_order"],
            "along_wall_offset": round(node["along"], 3),
            "center_z_offset": round(float(raw.get("center_z_offset") or 0.0), 3),
            "parent_id": node_ids[p_idx] if p_idx is not None else None,
            "parent_ref": raw.get("parent_ref"),
            "attach_slot": node.get("attach_slot"),
            "xm": round(node["xm"], 3),
            "zm": round(node["zm"], 3),
            "elevation_m": round(node["elev"], 3),
            "pos_x": round(node["xm"] / rw, 4),
            "pos_z": round(node["zm"] / rd, 4),
            "size_m": size_m,
            "world_box_m": [round(node["wx"], 2), round(node["fh"], 2), round(node["wz"], 2)],
            "rot_deg": int(node["rot_deg"]) % 360,
            "facing": raw.get("facing") or "",
            "facing_target": raw.get("facing_target") or "",
            "glb_fix_deg": 0.0,
            "glb_aligned_size": size_m,
            "is_glass": bool(raw.get("is_glass")),
            "confidence": round(float(raw.get("confidence") or 0.6), 2),
            "views": int(raw.get("views") or len(raw.get("seen_in_photos") or [1])),
            "seen_in_photos": raw.get("seen_in_photos") or [pidx],
            "lux3d_prompt": raw.get("appearance") or str(raw.get("label") or ""),
            "crop_rel": crop_rel,
            "crop_url": f"/api/pocketnest-v2/files/{proj['id']}/{crop_rel}" if crop_rel else None,
            "clean_source_rel": ctx_rel,
            "clean_source_url": f"/api/pocketnest-v2/files/{proj['id']}/{ctx_rel}" if ctx_rel else None,
            "clean_rel": None,
            "clean_url": None,
            "clean_error": None,
            "lux_status": "pending",
            "lux_task_id": None,
            "lux_version": None,
            "glb_rel": None,
            "glb_url": None,
        })

    spec["furniture"] = items
    spec["subzone_split"] = subzone_split
    spec["photo_orientations"] = orientations
    spec["audit"] = {
        "notes": (audit.get("notes") or [])[:10],
        "pass1_count": len(spec.get("_pass1_count") or []),
        "merged_count": len(merged),
    }
    return spec
