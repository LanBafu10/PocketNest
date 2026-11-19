#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""户型图像素级校准（PocketNest v3.1）。

解决两类「VLM bbox 不可全信」的问题：
1. 房间框外溢到纯白背景：按「墙体/地面前景像素」逐边裁掉白边；
2. 门洞位置不准：在 bbox 边缘找到真正的墙线，沿墙线找非墙色缺口（门洞），
   用缺口中心/宽度反算归一化 offset 与米制门宽；门被 VLM 标错墙时，
   在其他三面墙上找宽度合理的缺口并迁过去。

全部基于通用图像规则，不含任何房型/尺寸特例。
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _load(fp_abs: str):
    with Image.open(fp_abs) as im:
        im = im.convert("RGB")
        a = np.asarray(im).astype(np.int16)
    W, H = im.size
    mx, mn = a.max(2), a.min(2)
    mean = a.mean(2)
    fg = ((mx - mn) > 16) | (mean < 238)          # 彩色地面/墙线；纯白背景为 False
    dark = mean < 95                               # 黑色墙线/文字
    return fg, dark, W, H


def _trim_side(mask: np.ndarray, x1: int, y1: int, x2: int, y2: int, side: str,
               band_frac: float = 0.014, max_frac: float = 0.22, min_fg: float = 0.018):
    """把一条边向内收，直到边缘带里出现足够多的前景像素。"""
    W, H = mask.shape[1], mask.shape[0]
    if side in ("north", "south"):
        span = max(1, x2 - x1)
        band = max(2, int(span * band_frac))
        limit = int((y2 - y1) * max_frac)
        moved = 0
        while moved < limit:
            y = y1 if side == "north" else y2 - 1
            if y < 0 or y >= H:
                break
            frac = float(mask[y, x1:x2].mean()) if x2 > x1 else 1.0
            if frac >= min_fg:
                break
            if side == "north":
                y1 += 1
            else:
                y2 -= 1
            moved += 1
    else:
        span = max(1, y2 - y1)
        band = max(2, int(span * band_frac))
        limit = int((x2 - x1) * max_frac)
        moved = 0
        while moved < limit:
            x = x1 if side == "west" else x2 - 1
            if x < 0 or x >= W:
                break
            frac = float(mask[y1:y2, x].mean()) if y2 > y1 else 1.0
            if frac >= min_fg:
                break
            if side == "west":
                x1 += 1
            else:
                x2 -= 1
            moved += 1
    return x1, y1, x2, y2


def trim_rooms_to_content(rooms: list, fp_abs: str) -> list:
    fg, dark, W, H = _load(fp_abs)
    # 全局内容框（去掉四周纯白留白），只用于防止极端外溢
    cols = np.where(fg.sum(0) > max(3, H * 0.004))[0]
    rows = np.where(fg.sum(1) > max(3, W * 0.004))[0]
    gx1, gx2 = (int(cols[0]), int(cols[-1]) + 1) if len(cols) else (0, W)
    gy1, gy2 = (int(rows[0]), int(rows[-1]) + 1) if len(rows) else (0, H)

    for r in rooms:
        bb = r.get("bbox_1000")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x1 = int(max(gx1, min(W - 1, bb[0] / 1000.0 * W)))
        x2 = int(max(x1 + 8, min(gx2, bb[2] / 1000.0 * W)))
        y1 = int(max(gy1, min(H - 1, bb[1] / 1000.0 * H)))
        y2 = int(max(y1 + 8, min(gy2, bb[3] / 1000.0 * H)))
        for side in ("north", "south", "west", "east"):
            x1, y1, x2, y2 = _trim_side(fg, x1, y1, x2, y2, side)
        r["bbox_1000"] = [round(x1 / W * 1000, 1), round(y1 / H * 1000, 1),
                          round(x2 / W * 1000, 1), round(y2 / H * 1000, 1)]
    return rooms


def _find_gaps_on_wall(dark: np.ndarray, bbox_px: tuple, wall: str, gap_min: int = 12, gap_max: int = 70):
    """在 bbox 边缘 ±14px 的垂直带里统计墙线暗像素，返回低暗度缺口。

    单像素扫描会被门扇线/JPEG 噪点打断，带内求和更稳。
    """
    x1, y1, x2, y2 = bbox_px
    H, W = dark.shape
    band = 14
    if wall in ("north", "south"):
        base = y1 if wall == "north" else y2
        ys = list(range(max(0, base - band), min(H, base + band + 1)))
        xs = list(range(max(0, x1 + 2), min(W, x2 - 2)))
        if len(xs) < 10:
            return []
        cnt = np.array([dark[ys, x].sum() for x in xs])
        pos0, lo_ok, hi_ok = xs[0], x1 + 4, x2 - 4
    else:
        base = x1 if wall == "west" else x2
        xs = list(range(max(0, base - band), min(W, base + band + 1)))
        ys = list(range(max(0, y1 + 2), min(H, y2 - 2)))
        if len(ys) < 10:
            return []
        cnt = np.array([dark[y, xs].sum() for y in ys])
        pos0, lo_ok, hi_ok = ys[0], y1 + 4, y2 - 4

    if float(cnt.max()) < 2:
        return []
    thr = max(2.0, float(np.median(cnt)) * 0.35)
    gaps, run = [], 0
    for i, v in enumerate(np.append(cnt, 1e9)):
        if v <= thr:
            run += 1
        else:
            if gap_min <= run <= gap_max:
                a, b = pos0 + i - run, pos0 + i
                if a >= lo_ok and b <= hi_ok:
                    gaps.append((a, b))
            run = 0
    return gaps


def apply_wall_gaps(rooms: list, fp_abs: str, scale: float, img_w: int, img_h: int) -> list:
    """用墙线缺口校正每个 opening 的 wall / offset / width_m。"""
    fg, dark, W, H = _load(fp_abs)
    for r in rooms:
        bb = r.get("bbox_1000")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        x1 = int(bb[0] / 1000.0 * W)
        y1 = int(bb[1] / 1000.0 * H)
        x2 = int(bb[2] / 1000.0 * W)
        y2 = int(bb[3] / 1000.0 * H)
        bbox_px = (x1, y1, x2, y2)
        gaps = {w: _find_gaps_on_wall(dark, bbox_px, w) for w in ("north", "south", "west", "east")}
        used = {w: set() for w in gaps}

        openings = r.get("openings") or []
        unmatched = []

        def _plaus(kind, width_m):
            if kind == "window":
                return 0.45 <= width_m <= 3.2
            if kind == "sliding_door":
                return 0.85 <= width_m <= 3.0
            if kind == "open_arch":
                return 0.9 <= width_m <= 3.2
            return 0.50 <= width_m <= 1.35

        def _apply_gap(op, wall, a, b):
            span_px = max(1, (x2 - x1) if wall in ("north", "south") else (y2 - y1))
            wall_off_px = x1 if wall in ("north", "south") else y1
            op["wall"] = wall
            op["offset"] = round(max(0.06, min(0.94, ((a + b) / 2 - wall_off_px) / span_px)), 3)
            gw = (b - a) * scale
            kind = op.get("kind")
            if kind == "window":
                op["width_m"] = round(max(0.6, min(3.0, gw)), 2)
            elif kind in ("door", "entry_door"):
                op["width_m"] = round(max(0.66, min(1.15, gw)), 2)
            elif kind == "sliding_door":
                op["width_m"] = round(max(1.0, min(2.6, gw)), 2)
            else:
                op["width_m"] = round(max(1.0, min(3.0, gw)), 2)

        # Pass 1: 先锁定各自原本墙面上已经匹配到的缺口，避免跨墙迁移抢占其它门的缺口
        for op in openings:
            wall = op.get("wall")
            kind = op.get("kind")
            if wall not in gaps:
                continue
            span_px = max(1, (x2 - x1) if wall in ("north", "south") else (y2 - y1))
            wall_off_px = x1 if wall in ("north", "south") else y1
            center_px = wall_off_px + float(op.get("offset", 0.5)) * span_px

            best_i, best_d = None, None
            for i, (a, b) in enumerate(gaps[wall]):
                if i in used[wall] or not _plaus(kind, (b - a) * scale):
                    continue
                d = abs((a + b) / 2 - center_px)
                if d <= span_px * 0.40 and (best_d is None or d < best_d):
                    best_i, best_d = i, d

            if best_i is not None:
                used[wall].add(best_i)
                a, b = gaps[wall][best_i]
                _apply_gap(op, wall, a, b)
            else:
                unmatched.append(op)

        # Pass 2: 仅对非入户门、且原本墙面无缺口的门洞，尝试迁移到尚有未占用缺口的邻墙
        for op in unmatched:
            wall = op.get("wall")
            kind = op.get("kind")
            if kind not in ("door", "sliding_door", "open_arch"):
                continue
            span_px = max(1, (x2 - x1) if wall in ("north", "south") else (y2 - y1))
            wall_off_px = x1 if wall in ("north", "south") else y1
            center_px = wall_off_px + float(op.get("offset", 0.5)) * span_px
            for w2 in ("north", "south", "west", "east"):
                if w2 == wall:
                    continue
                cands = []
                for i, (a, b) in enumerate(gaps[w2]):
                    if i in used[w2]:
                        continue
                    if _plaus(kind, (b - a) * scale):
                        cands.append((i, abs(((a + b) / 2) - center_px)))
                if cands:
                    cands.sort(key=lambda t: t[1])
                    i = cands[0][0]
                    used[w2].add(i)
                    a, b = gaps[w2][i]
                    _apply_gap(op, w2, a, b)
                    break
    return rooms


def lost_corridor(large_room: dict, small_room: dict):
    """若大房间被小房间裁掉一条 >1.2m² 且宽 >0.7m 的连通条带，返回条带 bounds。

    用于把 L 形客厅 + 过道拆成两个矩形，而不是把走廊裁没。
    """
    lb = large_room.get("_bounds")
    sb = small_room.get("_bounds")
    if not lb or not sb:
        return None
    ox = min(lb[2], sb[2]) - max(lb[0], sb[0])
    oz = min(lb[3], sb[3]) - max(lb[1], sb[1])
    if ox <= 0.05 or oz <= 0.05:
        return None
    # 小房间整体在大的 bbox 内部，且大房间在某一侧还有连续条带
    if sb[0] > lb[0] + 0.7 and ox / max(0.1, lb[2] - lb[0]) < 0.55:
        strip = [sb[2], max(lb[1], sb[1]), lb[2], min(lb[3], sb[3])]
    elif sb[1] > lb[1] + 0.7 and oz / max(0.1, lb[3] - lb[1]) < 0.55:
        strip = [max(lb[0], sb[0]), sb[3], min(lb[2], sb[2]), lb[3]]
    else:
        return None
    w, d = strip[2] - strip[0], strip[3] - strip[1]
    if w >= 0.7 and d >= 0.7 and w * d >= 1.2:
        return strip
    return None


def _wall_lines(dark: np.ndarray, axis: str, thr_frac: float = 0.08, min_len: int = 24):
    """返回强墙线 [(中心坐标, 强度)]；axis='x' 找竖墙，'y' 找横墙。"""
    H, W = dark.shape
    counts = dark.sum(0) if axis == "x" else dark.sum(1)
    limit = (H if axis == "x" else W) * thr_frac
    idx = [i for i, c in enumerate(counts) if c > limit]
    groups, cur = [], []
    for i in idx:
        if cur and i - cur[-1] <= 3:
            cur.append(i)
        else:
            if cur:
                groups.append(cur)
            cur = [i]
    if cur:
        groups.append(cur)
    lines = []
    for g in groups:
        if len(g) < 2 and max(counts[i] for i in g) < min_len:
            continue
        center = sum(g) / len(g)
        strength = float(max(counts[i] for i in g))
        lines.append((center, strength))
    return lines


def _local_wall_support(dark: np.ndarray, axis: str, pos: float, span_lo: float, span_hi: float) -> float:
    """检查在垂直区间 [span_lo, span_hi] 上，pos 附近 (±4px) 是否真的有墙线暗像素支撑。"""
    H, W = dark.shape
    p = int(round(pos))
    lo = int(round(span_lo))
    hi = int(round(span_hi))
    if hi - lo < 16:
        return 0.0
    # 去掉两端 8% 避免角点误触其他墙
    pad = max(2, int((hi - lo) * 0.08))
    lo2, hi2 = lo + pad, hi - pad
    if hi2 <= lo2:
        lo2, hi2 = lo, hi
    if axis == "x":
        x0, x1 = max(0, p - 4), min(W, p + 5)
        y0, y1 = max(0, lo2), min(H, hi2)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        col_hit = dark[y0:y1, x0:x1].any(axis=1)
        return float(col_hit.mean())
    else:
        y0, y1 = max(0, p - 4), min(H, p + 5)
        x0, x1 = max(0, lo2), min(W, hi2)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        row_hit = dark[y0:y1, x0:x1].any(axis=0)
        return float(row_hit.mean())


def snap_rooms_to_walls(rooms: list, fp_abs: str, make_corridor: bool = True) -> list:
    """把 bbox 边缘吸到局部真实强墙线上；重叠房间共用同一道墙，并保留连通干区/过道。"""
    fg, dark, W, H = _load(fp_abs)
    vlines = _wall_lines(dark, "x")
    hlines = _wall_lines(dark, "y")

    def to_px(bb):
        return [bb[0] / 1000.0 * W, bb[1] / 1000.0 * H, bb[2] / 1000.0 * W, bb[3] / 1000.0 * H]

    max_snap_d = min(68.0, 0.045 * max(W, H))

    def nearest_supported(lines, axis: str, v: float, span_lo: float, span_hi: float):
        best, best_score = None, -1.0
        for pos, _ in lines:
            d = abs(pos - v)
            if d > max_snap_d:
                continue
            sup = _local_wall_support(dark, axis, pos, span_lo, span_hi)
            # 远距离吸附（>42px）必须有非常强的局部实墙支撑（>=0.42），避免错吸到邻室墙线
            req_sup = 0.42 if d > min(44.0, 0.030 * max(W, H)) else 0.28
            if sup < req_sup:
                continue
            score = sup * 1.4 - (d / max(1.0, max_snap_d)) * 0.6
            if score > best_score:
                best, best_score = pos, score
        return best

    px = {r["name"]: to_px(r["bbox_1000"]) for r in rooms}
    # 1) 单边局部支撑吸附：只吸有局部墙线支撑的强墙，防止跨房吸到隔壁卧室的错位墙
    for name, b in px.items():
        new_b = list(b)
        p0 = nearest_supported(vlines, "x", b[0], b[1], b[3])
        p2 = nearest_supported(vlines, "x", b[2], b[1], b[3])
        p1 = nearest_supported(hlines, "y", b[1], b[0], b[2])
        p3 = nearest_supported(hlines, "y", b[3], b[0], b[2])
        if p0 is not None and (p2 is None or p2 - p0 >= 36):
            new_b[0] = p0
        if p2 is not None and p2 - new_b[0] >= 36:
            new_b[2] = p2
        if p1 is not None and (p3 is None or p3 - p1 >= 36):
            new_b[1] = p1
        if p3 is not None and p3 - new_b[1] >= 36:
            new_b[3] = p3
        if new_b[2] - new_b[0] < 36:
            new_b[2] = new_b[0] + 36
        if new_b[3] - new_b[1] < 36:
            new_b[3] = new_b[1] + 36
        px[name] = new_b

    # 2) 重叠房间：包含/L形环绕关系优先裁大房间，并保留被裁掉的干湿过道/玄关条带
    names = list(px.keys())
    extra = []
    has_alcove_already = any(
        r.get("category") in ("entry", "other") or any(k in r.get("name", "") for k in ("过道", "干区", "走廊", "玄关"))
        for r in rooms
    )
    for _ in range(6):
        changed = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = px[names[i]], px[names[j]]
                ox = min(a[2], b[2]) - max(a[0], b[0])
                oz = min(a[3], b[3]) - max(a[1], b[1])
                if ox <= 2 or oz <= 2:
                    continue
                area_a = max(1.0, (a[2] - a[0]) * (a[3] - a[1]))
                area_b = max(1.0, (b[2] - b[0]) * (b[3] - b[1]))
                small, large = (a, b) if area_a <= area_b else (b, a)
                small_name = names[j] if area_a <= area_b else names[i]
                large_name = names[i] if area_a <= area_b else names[j]
                small_area = min(area_a, area_b)
                if (ox * oz) / max(1.0, small_area) > 0.24:
                    # 选保留 large 最大剩余面积的裁剪方向（避免把 L 形客厅主躯干切没）
                    cands_trim = []
                    if small[0] > large[0] + 36:
                        cands_trim.append(("right_cut", (small[0] - large[0]) * (large[3] - large[1]), [large[0], large[1], small[0], large[3]]))
                    if large[2] > small[2] + 36:
                        cands_trim.append(("left_cut", (large[2] - small[2]) * (large[3] - large[1]), [small[2], large[1], large[2], large[3]]))
                    if small[1] > large[1] + 36:
                        cands_trim.append(("bot_cut", (large[2] - large[0]) * (small[1] - large[1]), [large[0], large[1], large[2], small[1]]))
                    if large[3] > small[3] + 36:
                        cands_trim.append(("top_cut", (large[2] - large[0]) * (large[3] - small[3]), [large[0], small[3], large[2], large[3]]))

                    large_name = names[i] if large is a else names[j]
                    small_name = names[j] if large is a else names[i]
                    large_cat = next((q.get("category") for q in rooms if q["name"] == large_name), "")
                    small_cat = next((q.get("category") for q in rooms if q["name"] == small_name), "")

                    # 在被裁掉的翼区里查找属于室内前景（非纯白背景）的连通条带保留为「过道」
                    if make_corridor and not has_alcove_already and large_cat in ("living", "dining") and small_cat in ("bath", "kitchen"):
                        wing_strips = []
                        if small[3] < large[3] - 40 and small[0] >= large[0] + 18:
                            wing_strips.append(("west", [max(small[0], large[0]), small[3], min(small[2], large[2]), large[3]]))
                        if small[1] > large[1] + 40 and small[0] >= large[0] + 18:
                            wing_strips.append(("west", [max(small[0], large[0]), large[1], min(small[2], large[2]), small[1]]))
                        if small[0] > large[0] + 40 and small[1] >= large[1] + 18:
                            wing_strips.append(("north", [large[0], max(small[1], large[1]), small[0], min(small[3], large[3])]))
                        for conn_wall, strip in wing_strips:
                            sw_px, sh_px = strip[2] - strip[0], strip[3] - strip[1]
                            if sw_px < 36 or sh_px < 36 or sw_px * sh_px < 1400:
                                continue
                            # 必须是真实室内区域（fg 前景占比 >= 0.45，排除屋外白底）
                            sx0, sy0 = max(0, int(strip[0])), max(0, int(strip[1]))
                            sx1, sy1 = min(W, int(strip[2])), min(H, int(strip[3]))
                            if sx1 <= sx0 or sy1 <= sy0 or float(fg[sy0:sy1, sx0:sx1].mean()) < 0.45:
                                continue
                            overlaps_other = False
                            for other_name, ob in px.items():
                                if other_name == large_name:
                                    continue
                                if min(strip[2], ob[2]) - max(strip[0], ob[0]) > 6 and min(strip[3], ob[3]) - max(strip[1], ob[1]) > 6:
                                    overlaps_other = True
                                    break
                            if overlaps_other:
                                continue
                            name = "过道"
                            k = 1
                            while name in px:
                                k += 1
                                name = f"过道{k}"
                            px[name] = strip
                            extra.append({"name": name, "category": "entry", "bbox_1000": [
                                round(strip[0] / W * 1000, 1), round(strip[1] / H * 1000, 1),
                                round(strip[2] / W * 1000, 1), round(strip[3] / H * 1000, 1)],
                                "est_width_m": 0, "est_depth_m": 0, "window_dir": "NONE",
                                "openings": [{"kind": "open_passage", "wall": conn_wall, "offset": 0.5,
                                              "width_m": round(max(0.75, min(1.4, (strip[3] - strip[1]) / max(1.0, H) * 12.0)), 2),
                                              "target": large_name, "render_leaf": False}]})
                            names.append(name)
                            has_alcove_already = True
                            break

                    if cands_trim:
                        cands_trim.sort(key=lambda t: t[1], reverse=True)
                        large[:] = cands_trim[0][2]
                    elif ox <= oz:
                        if (small[0] + small[2]) / 2 >= (large[0] + large[2]) / 2:
                            large[2] = min(large[2], small[0])
                        else:
                            large[0] = max(large[0], small[2])
                    else:
                        if (small[1] + small[3]) / 2 >= (large[1] + large[3]) / 2:
                            large[3] = min(large[3], small[1])
                        else:
                            large[1] = max(large[1], small[3])
                    changed = True
                    continue
                if ox <= oz:
                    acx, bcx = (a[0] + a[2]) / 2, (b[0] + b[2]) / 2
                    lo, hi = max(a[0], b[0]), min(a[2], b[2])
                    py0, py1 = max(a[1], b[1]), min(a[3], b[3])
                    cands = [
                        (p, _local_wall_support(dark, "x", p, py0, py1))
                        for p, _ in vlines if lo - 8 <= p <= hi + 8
                    ]
                    cands = [(p, s) for p, s in cands if s >= 0.20]
                    wall = max(cands, key=lambda t: t[1])[0] if cands else (lo + hi) / 2.0
                    if acx <= bcx:
                        a[2], b[0] = wall, wall
                    else:
                        a[0], b[2] = wall, wall
                else:
                    acy, bcy = (a[1] + a[3]) / 2, (b[1] + b[3]) / 2
                    lo, hi = max(a[1], b[1]), min(a[3], b[3])
                    px0, px1 = max(a[0], b[0]), min(a[2], b[2])
                    cands = [
                        (p, _local_wall_support(dark, "y", p, px0, px1))
                        for p, _ in hlines if lo - 8 <= p <= hi + 8
                    ]
                    cands = [(p, s) for p, s in cands if s >= 0.20]
                    wall = max(cands, key=lambda t: t[1])[0] if cands else (lo + hi) / 2.0
                    if acy <= bcy:
                        a[3], b[1] = wall, wall
                    else:
                        a[1], b[3] = wall, wall
                changed = True
        if not changed:
            break

    if extra:
        rooms.extend(extra)

    for r in rooms:
        b = px[r["name"]]
        # 仅当不会撞到其它房间时，微调过薄房间（< ~52px）
        for i, j, lim in ((0, 2, W), (1, 3, H)):
            if b[j] - b[i] < 48:
                mid = (b[i] + b[j]) / 2
                cand0, cand1 = max(0.0, mid - 24), min(float(lim), mid + 24)
                # 只在不产生新重叠时扩张
                ok = True
                for r2 in rooms:
                    if r2["name"] == r["name"]:
                        continue
                    ob = px[r2["name"]]
                    if i == 0:
                        if min(cand1, ob[2]) - max(cand0, ob[0]) > 2 and min(b[3], ob[3]) - max(b[1], ob[1]) > 2:
                            ok = False; break
                    else:
                        if min(b[2], ob[2]) - max(b[0], ob[0]) > 2 and min(cand1, ob[3]) - max(cand0, ob[1]) > 2:
                            ok = False; break
                if ok:
                    b[i], b[j] = cand0, cand1
        r["bbox_1000"] = [round(max(0, b[0]) / W * 1000, 1), round(max(0, b[1]) / H * 1000, 1),
                          round(min(W, b[2]) / W * 1000, 1), round(min(H, b[3]) / H * 1000, 1)]
    return rooms

