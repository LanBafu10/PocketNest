#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多角度 3D 截图 + 实景照片真实视觉对照自纠偏闭环（PocketNest v3.1）。

流程：
1. headless Chromium 打开 /pocketnest/?pid=..
   - 干净模式（无 debug 箭头）：top_clean / iso_clean / 每个主要房间 3D 视角 clean_{rid}
   - 调试模式（debug_pose=1）：top / iso / 每个房间 0/90/180/270 四个方位特写（房间外隐藏）
2. 每个房间把 [户型图 + top(debug) + iso(clean) + 四方位特写 + 房间实景照片(最多3张)] 喂给 VLM
   （默认 gemini-4-hydrogen-high, max_tokens=16384），真图实景对照检查：
   - 双门/重叠/堵门问题
   - 餐椅是否围绕餐桌朝桌、书桌椅是否朝书桌
   - 床/衣柜/书桌/电视/沙发/冰箱/橱柜/马桶/洗衣机/阳台柜是否靠对真墙且正面朝内
   - 连通阳台是否与主区正确分区（洗衣机/阳台柜在阳台，床不在阳台）
3. 高置信度 rot_deg / facing patch 自动回写 project.json 并复验，最多 max_rounds 轮。

CLI: python3 pose_audit.py <pid> [max_rounds]
"""

from __future__ import annotations

import json
import os
import time

from PIL import Image

import pipeline_common as pc

PUBLIC = os.environ.get("PN_PUBLIC_URL", "http://127.0.0.1:8080/pocketnest/")
AUDIT_MODEL = os.environ.get("PN_POSE_MODEL", pc.AUDIT_VLM_MODEL)


def _save_png_and_jpg(page, png_path: str, write_png: bool = True):
    jpg_path = os.path.splitext(png_path)[0] + ".jpg"
    if write_png:
        page.screenshot(path=png_path, full_page=False, timeout=120000)
        try:
            with Image.open(png_path) as im:
                im.convert("RGB").save(jpg_path, format="JPEG", quality=90)
        except Exception:
            pass
        return png_path
    page.screenshot(path=jpg_path, type="jpeg", quality=88, full_page=False, timeout=120000)
    return jpg_path


def _capture(pid: str, outdir: str, rooms: list) -> dict:
    from playwright.sync_api import sync_playwright

    os.makedirs(outdir, exist_ok=True)
    shots = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(args=[
            "--no-sandbox", "--disable-dev-shm-usage", "--enable-unsafe-swiftshader",
            "--use-gl=swiftshader", "--js-flags=--max-old-space-size=4096",
        ])
        ctx = browser.new_context(viewport={"width": 1280, "height": 860})

        # 1) 干净高清截图（无 debug_pose 箭头/罗盘）
        pg_clean = ctx.new_page()
        pg_clean.goto(f"{PUBLIC}?pid={pid}&view=top", wait_until="load", timeout=90000)
        try:
            pg_clean.wait_for_function("window.__poseReady === true", timeout=90000)
        except Exception as e:
            browser.close()
            raise RuntimeError(f"3D 干净场景 90s 未就绪，拒绝截空图：{e}")
        pg_clean.wait_for_timeout(900)

        def shoot_clean(name):
            path = os.path.join(outdir, name + ".png")
            _save_png_and_jpg(pg_clean, path)
            shots[name] = path

        shoot_clean("top_clean")
        pg_clean.evaluate("window.__setPoseView('iso','',0)")
        try:
            pg_clean.wait_for_function("window.__poseReady === true", timeout=60000)
        except Exception:
            pass
        pg_clean.wait_for_timeout(500)
        shoot_clean("iso_clean")

        for r in rooms:
            pg_clean.evaluate("([v, rid, y]) => window.__setPoseView(v, rid, y)", ["room", r["id"], 35])
            try:
                pg_clean.wait_for_function("window.__poseReady === true", timeout=60000)
            except Exception:
                pass
            pg_clean.wait_for_timeout(450)
            shoot_clean(f"clean_{r['id']}")
        pg_clean.close()

        # 2) debug_pose=1 调试截图（带 N/E/S/W 罗盘、红/蓝/绿门窗条、家具正面红箭头）
        pg = ctx.new_page()
        pg.goto(f"{PUBLIC}?pid={pid}&debug_pose=1&view=top", wait_until="load", timeout=90000)
        try:
            pg.wait_for_function("window.__poseReady === true", timeout=90000)
        except Exception as e:
            browser.close()
            raise RuntimeError(f"3D 调试场景 90s 未就绪，拒绝截空图：{e}")
        pg.wait_for_timeout(900)

        def shoot(name, write_png=False):
            path = os.path.join(outdir, name + ".png")
            out_p = _save_png_and_jpg(pg, path, write_png=write_png)
            shots[name] = path if write_png else out_p

        shoot("top", write_png=True)
        pg.evaluate("window.__setPoseView('iso','',0)")
        try:
            pg.wait_for_function("window.__poseReady === true", timeout=60000)
        except Exception:
            pass
        pg.wait_for_timeout(500)
        shoot("iso", write_png=True)

        for r in rooms:
            for yaw in (0, 90, 180, 270):
                pg.evaluate("([v, rid, y]) => window.__setPoseView(v, rid, y)", ["room", r["id"], yaw])
                try:
                    pg.wait_for_function("window.__poseReady === true", timeout=60000)
                except Exception:
                    pass
                pg.wait_for_timeout(400)
                shoot(f"room_{r['id']}_yaw{yaw}")
        browser.close()
    return shots


def _compact_items(room: dict) -> str:
    rows = []
    for it in ((room.get("scene_spec") or {}).get("furniture") or []):
        rows.append({
            "id": it.get("id"), "label": it.get("label"), "type": it.get("type"),
            "zone": it.get("zone"),
            "rot_deg": it.get("rot_deg"), "wall_anchor": it.get("wall_anchor"),
            "along_wall_offset": it.get("along_wall_offset"),
            "xm": it.get("xm"), "zm": it.get("zm"),
            "facing": it.get("facing"), "facing_target": it.get("facing_target"),
            "parent_id": it.get("parent_id"), "attach_slot": it.get("attach_slot"),
            "size_m": it.get("size_m"), "elevation_m": it.get("elevation_m"),
        })
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":"))


def _prompt(room: dict, walls: str, n_real: int) -> str:
    split = (room.get("scene_spec") or {}).get("subzone_split") or {}
    subzone_txt = f"\n【连通阳台分区】{json.dumps(split, ensure_ascii=False)}" if split.get("has_connected_balcony") else ""
    return f"""你是 3D 空间建模与实景孪生的高级验收官。请将 3D 渲染截图与【原始户型图 + 实景真图】做严格逐项对比，绝不放水：
- 第 1 张：原始户型图。
- 第 2~3 张：全屋 3D 俯视图（top，带 N/E/S/W 罗盘与红箭头）与干净全屋等轴测图（iso_clean）。
- 第 4~7 张：房间【{room['name']}】的 0°/90°/180°/270° 四方位 3D 调试特写（红条=门洞、蓝条=窗、绿条=垭口；每件家具中心的**红色箭头 = 家具正面本地 +Z**）。
- 最后 {n_real} 张：房间【{room['name']}】的**实景真图照片**。

【户型图锁定的四面墙与门窗】
{walls}{subzone_txt}

【本房间家具清单】
{_compact_items(room)}

【严格视觉对比验收清单】
1. **门洞与房间重叠**：是否出现同一门洞两扇重叠门（双门）？房间之间有没有穿墙/重叠？门前是否被家具堵死？
2. **实景真图对比（重点！）**：对比最后几张实景真图，家具所在的墙面方位（N/E/S/W 左右位置）、靠墙贴合度、主家具与从属件关系是否一致？
   - 餐厅/客厅：餐椅是否整齐围绕餐桌且正面（红箭头）朝向餐桌？还是散乱/背对？沙发、冰箱、茶几是否摆在正确位置且不堵门？
   - 卧室/书房：床是否放在主卧区（绝不可掉进连通阳台！），床头靠真墙、床尾红箭头朝内？衣柜、书桌、书桌椅（朝向书桌）、电视/支架是否分别靠真墙且正面朝房间内（不可面壁！）？若有连通阳台（绿条/窗帘垭口分隔），洗衣机/阳台柜是否在阳台区？
   - 厨卫：橱柜/吊柜/马桶/洗手台/热水器/花洒是否紧贴真墙、正面朝房间可用空间？
3. 若红色箭头方向错了，给出需要旋转的角度：只允许 +90 / -90 / 180（顺时针为正）。
   - 若家具**靠错了墙面**或**沿墙左右/上下位置明显偏错**，请直接在 `corrections` 中同时给出修正后的 `wall_anchor`（`"north"|"south"|"west"|"east"|"center"|"corner_nw"|"corner_ne"|"corner_sw"|"corner_se"`）与 `along_wall_offset`（`-1.0~+1.0`），并给出 `confidence >= 0.85`，系统会自动调用约束求解器重算该房间布局。
4. **只根据图像真实内容判断**：若存在跨房间结构性门窗错误或缺失模型（光靠 `corrections` 调位修不好），请在 `door_issues` / `layout_issues` / `summary` 明确指出，且必须把 `pass` 设为 `false`！
5. 只有所有家具位置、墙面、朝向、门洞全部正确且符合实景真图时才可 `pass=true`；有任何疑问一律 `pass=false`。

严格返回 JSON：
{{
  "pass": false,
  "corrections": [
    {{"item_id": "item_xxx", "delta_deg": 90, "facing": "table", "facing_target": "餐桌",
      "wall_anchor": null, "along_wall_offset": null,
      "confidence": 0.0, "reason": "简短说明"}}
  ],
  "door_issues": [
    {{"wall": "north", "offset": 0.78, "width_m": 0.9, "reason": "门洞问题说明"}}
  ],
  "layout_issues": ["位置/墙面/分区/遮挡问题说明（如有）"],
  "summary": "整体对照结论"
}}"""


def _apply(pid: str, patches: list, round_no: int) -> int:
    proj = pc.load_project(pid)
    if not proj:
        return 0
    by_id = {}
    for r in proj.get("rooms", []):
        for it in ((r.get("scene_spec") or {}).get("furniture") or []):
            by_id[it.get("id")] = (r, it)
    changed = 0
    for c in patches:
        if not isinstance(c, dict):
            continue
        pair = by_id.get(str(c.get("item_id")))
        if not pair:
            continue
        r, it = pair
        try:
            conf = float(c.get("confidence") or 0)
        except Exception:
            conf = 0
        if conf < 0.70:
            continue
        WALL_INWARD = {"north": 0, "south": 180, "west": 90, "east": 270}
        WALL_LOCKED_TYPES = {
            "bed", "sofa", "wardrobe", "fridge", "washer", "toilet",
            "kitchen_counter", "wall_cabinet", "tv", "tv_stand", "desk",
            "drying_rack", "vanity_sink",
        }
        anchor = str(it.get("wall_anchor") or "")
        delta = c.get("delta_deg")
        try:
            delta = int(delta)
        except Exception:
            delta = 0
        if delta in (90, -90, 180, -180, 270, -270):
            new_rot = (int(it.get("rot_deg") or 0) + delta) % 360
            # 单墙锚定主件或挂墙件绝不允许被旋转成面壁或侧向（应锁死朝室内）
            if anchor in WALL_INWARD and (it.get("type") in WALL_LOCKED_TYPES or str(it.get("mount") or "") in ("wall", "wall_high")) and new_rot != WALL_INWARD[anchor]:
                continue
            it["rot_deg"] = new_rot
            # 若旋转了 ±90°，同步互换 world_box_m 并夹紧在房间边界内防止穿墙
            if abs(delta) in (90, 270):
                sz = it.get("size_m") or [0.8, 0.8, 0.6]
                fw, fh, fd = float(sz[0]), float(sz[1]), float(sz[2])
                wx = fw if new_rot in (0, 180) else fd
                wz = fd if new_rot in (0, 180) else fw
                it["world_box_m"] = [round(wx, 2), round(fh, 2), round(wz, 2)]
                rw, rd = float(r.get("width_m") or 3.4), float(r.get("depth_m") or 3.4)
                pad = 0.07
                it["xm"] = round(max(-rw / 2 + wx / 2 + pad, min(rw / 2 - wx / 2 - pad, float(it.get("xm") or 0))), 3)
                it["zm"] = round(max(-rd / 2 + wz / 2 + pad, min(rd / 2 - wz / 2 - pad, float(it.get("zm") or 0))), 3)
            changed += 1
        if c.get("facing") in ("room_center", "table", "dining_table", "coffee_table",
                               "sofa", "bed", "tv", "tv_stand", "desk", "north", "south", "west", "east"):
            if c["facing"] != it.get("facing"):
                it["facing"] = c["facing"]
                changed += 1
        if c.get("facing_target"):
            it["facing_target"] = str(c["facing_target"])[:22]
        suggestion = {}
        need_relayout = False
        if c.get("wall_anchor") in ("north", "south", "west", "east", "center",
                                    "corner_nw", "corner_ne", "corner_sw", "corner_se"):
            suggestion["wall_anchor"] = c["wall_anchor"]
            if conf >= 0.85 and c["wall_anchor"] != it.get("wall_anchor"):
                it["wall_anchor"] = c["wall_anchor"]
                if c["wall_anchor"] in WALL_INWARD:
                    it["rot_deg"] = WALL_INWARD[c["wall_anchor"]]
                need_relayout = True
                changed += 1
        if c.get("along_wall_offset") is not None:
            suggestion["along_wall_offset"] = c["along_wall_offset"]
            try:
                ao = float(c["along_wall_offset"])
                if conf >= 0.85 and abs(ao - float(it.get("along_wall_offset") or 0.0)) > 0.12:
                    it["along_wall_offset"] = round(max(-1.0, min(1.0, ao)), 3)
                    need_relayout = True
                    changed += 1
            except Exception:
                pass
        it["pose_audit"] = {"round": round_no, "reason": str(c.get("reason") or "")[:160],
                            "confidence": conf, "ts": int(time.time()),
                            "suggestion": suggestion or None}
        if need_relayout:
            from pipeline_room import re_solve_existing_room
            re_solve_existing_room(r)
    if changed:
        pc.save_project(pid, proj)
    return changed


def programmatic_fix(pid: str) -> dict:
    """不依赖 VLM 的确定性朝向自检：餐椅朝桌、书桌椅朝桌、贴墙件正面离墙、轴对齐件不斜放，并重跑布局校验。"""
    proj = pc.load_project(pid)
    if not proj:
        return {"changes": []}
    import math
    from pipeline_room import recompute_world_box, re_solve_existing_room

    def ang(dx, dz):
        return int(round(math.degrees(math.atan2(dx, dz)))) % 360

    def diff(a, b):
        return abs((a - b + 180) % 360 - 180)

    changes = []
    for room in proj.get("rooms", []):
        items = (room.get("scene_spec") or {}).get("furniture") or []
        tables = [i for i in items if i.get("type") in ("dining_table", "coffee_table")]
        desks = [i for i in items if i.get("type") == "desk"]
        for it in items:
            rot = int(it.get("rot_deg") or 0) % 360
            xm, zm = float(it.get("xm") or 0), float(it.get("zm") or 0)
            new = None
            if it.get("type") == "chair":
                parent = next((q for q in items if q.get("id") == it.get("parent_id")), None)
                if parent:
                    desired = ang(float(parent.get("xm") or 0) - xm, float(parent.get("zm") or 0) - zm)
                    if diff(rot, desired) > 30:
                        new = desired
                elif desks and "桌" in str(it.get("facing_target") or "") and "餐" not in str(it.get("label") or ""):
                    d0 = min(desks, key=lambda q: (float(q.get("xm") or 0) - xm) ** 2 + (float(q.get("zm") or 0) - zm) ** 2)
                    desired = ang(float(d0.get("xm") or 0) - xm, float(d0.get("zm") or 0) - zm)
                    if diff(rot, desired) > 35:
                        new = desired
                elif tables:
                    t = min(tables, key=lambda q: (float(q.get("xm") or 0) - xm) ** 2 + (float(q.get("zm") or 0) - zm) ** 2)
                    desired = ang(float(t.get("xm") or 0) - xm, float(t.get("zm") or 0) - zm)
                    if diff(rot, desired) > 35:
                        new = desired
            elif it.get("wall_anchor") in ("north", "south", "west", "east", "corner_nw", "corner_ne", "corner_sw", "corner_se"):
                want = {"north": 0, "corner_nw": 0, "corner_ne": 0,
                        "south": 180, "corner_sw": 180, "corner_se": 180,
                        "west": 90, "east": 270}[it["wall_anchor"]]
                if diff(rot, want) > 100:
                    new = want
                elif it.get("type") in ("wardrobe", "bookshelf", "tv_stand", "fridge", "desk", "washer",
                                        "toilet", "bathtub", "vanity_sink", "kitchen_counter", "wall_cabinet"):
                    snap = int(round(rot / 90.0)) * 90 % 360
                    if diff(rot, snap) > 20:
                        new = snap
            if new is not None and new != rot:
                it["rot_deg"] = new
                it["world_box_m"] = recompute_world_box(it.get("size_m"), new)
                it["pose_audit"] = {"round": 0, "reason": "确定性自检：餐椅/书桌椅朝桌/贴墙件正面离墙/轴对齐",
                                    "confidence": 1.0, "ts": int(time.time()), "was": rot}
                changes.append({"room": room["name"], "item_id": it.get("id"), "label": it.get("label"),
                                "from": rot, "to": new})
        # 用通用多锚点求解器统一刷新该房间内全部碰撞/净空/贴墙尺寸与朝向
        if items:
            re_solve_existing_room(room)
    pc.save_project(pid, proj)
    proj = pc.load_project(pid)
    if proj:
        pipe = proj.setdefault("pipeline", {})
        pipe["pose_programmatic"] = {"ts": int(time.time()), "changes": changes}
        pc.save_project(pid, proj)
    return {"changes": changes}


def run(pid: str, max_rounds: int = 2) -> dict:
    proj = pc.load_project(pid)
    if not proj:
        raise ValueError("项目不存在")
    programmatic_fix(pid)
    proj = pc.load_project(pid)
    pdir = pc.project_dir(pid)
    fp_url = ((proj.get("floorplan") or {}).get("url") or "")
    fp_abs = os.path.join(pdir, fp_url.split("/")[-1]) if fp_url else None
    outdir = os.path.join(pdir, "debug", "pose")
    rooms = [r for r in proj.get("rooms", [])
             if (r.get("scene_spec") or {}).get("furniture")]

    report = {"pid": pid, "model": AUDIT_MODEL, "rounds": [], "started_at": int(time.time()), "ok": False}
    last_round_ok = False
    for rnd in range(1, max_rounds + 1):
        try:
            shots = _capture(pid, outdir, rooms)
        except Exception as e:  # noqa: BLE001
            report["capture_error"] = str(e)[:300]
            report["finished_at"] = int(time.time())
            return report
        proj = pc.load_project(pid)
        total_changes = 0
        last_round_ok = True
        round_log = {"round": rnd, "shots": shots, "rooms": []}
        from concurrent.futures import ThreadPoolExecutor

        def _audit_one(room):
            r = next((x for x in proj.get("rooms", []) if x["id"] == room["id"]), None)
            if not r or not ((r.get("scene_spec") or {}).get("furniture")):
                return None
            walls = __import__("pipeline_room").describe_walls(r, proj.get("rooms", []))
            imgs = [fp_abs] if fp_abs and os.path.isfile(fp_abs) else []
            imgs += [shots.get("top"), shots.get("iso_clean")]
            for yaw in (0, 90, 180, 270):
                imgs.append(shots.get(f"room_{r['id']}_yaw{yaw}"))
            real_photos = [os.path.join(pdir, ph.get("rel_path", "")) for ph in (r.get("photos") or [])[:3]]
            real_photos = [p for p in real_photos if p and os.path.isfile(p)]
            imgs = [p for p in (imgs + real_photos) if p and os.path.isfile(p)][:10]
            try:
                res = pc.call_vlm_json(_prompt(r, walls, len(real_photos)), imgs, timeout=240, max_images=10,
                                       model=AUDIT_MODEL, max_tokens=16384)
                return (r, res, None)
            except Exception as e:  # noqa: BLE001
                return (r, None, str(e)[:200])

        with ThreadPoolExecutor(max_workers=min(2, max(1, len(rooms)))) as ex:
            results = list(ex.map(_audit_one, rooms))

        for item in results:
            if not item:
                continue
            r, res, err = item
            if err:
                last_round_ok = False
                round_log["rooms"].append({"room": r["name"], "error": err, "pass": False})
                continue
            patches = res.get("corrections") or []
            n = _apply(pid, patches, rnd)
            total_changes += n
            verdict = bool(res.get("pass"))
            if not verdict:
                last_round_ok = False
            round_log["rooms"].append({
                "room": r["name"], "pass": verdict, "patches": patches,
                "door_issues": res.get("door_issues") or [],
                "layout_issues": res.get("layout_issues") or [],
                "summary": res.get("summary"), "applied": n,
                "model": (res.get("_vlm") or {}).get("model"),
            })
        report["rounds"].append(round_log)
        if total_changes == 0:
            break

    proj = pc.load_project(pid)
    report["ok"] = bool(last_round_ok)
    if proj:
        pipe = proj.setdefault("pipeline", {})
        pipe["pose_audit"] = report
        pc.save_project(pid, proj)
    if report["rounds"] and any(sum(x.get("applied", 0) for x in rd.get("rooms", [])) > 0 for rd in report["rounds"]):
        try:
            shots = _capture(pid, outdir, [r for r in (pc.load_project(pid) or {}).get("rooms", [])
                                           if (r.get("scene_spec") or {}).get("furniture")])
            report["final_shots"] = shots
        except Exception:
            pass
    elif report["rounds"]:
        report["final_shots"] = report["rounds"][-1].get("shots")
    report["finished_at"] = int(time.time())
    return report


if __name__ == "__main__":
    import sys
    pid = sys.argv[1]
    rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    rep = run(pid, rounds)
    print(json.dumps(rep, ensure_ascii=False, indent=1)[:12000])
