#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""同步构建免登录静态公开演示站 /srv/pocketnest-demo/（含严格个人隐私脱敏）。"""

import json
import os
import shutil

SRC_WEB = "./frontend"
SRC_STORAGE = "./storage"
DEST_DEMO = "/srv/pocketnest-demo"

PIDS = ["nest_test_v5", "nest_case1", "nest_case2", "nest_case3"]


def copy_if_exists(src, dst):
    if os.path.isfile(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def main():
    if os.path.isdir(DEST_DEMO):
        shutil.rmtree(DEST_DEMO)
    os.makedirs(DEST_DEMO, exist_ok=True)

    # 1. 前端静态代码与 Three.js vendor
    shutil.copy2(os.path.join(SRC_WEB, "index.html"), os.path.join(DEST_DEMO, "index.html"))
    shutil.copytree(os.path.join(SRC_WEB, "assets"), os.path.join(DEST_DEMO, "assets"))
    shutil.copytree(os.path.join(SRC_WEB, "vendor"), os.path.join(DEST_DEMO, "vendor"))

    # 2. 拷贝并脱敏 4 套户型数据
    for pid in PIDS:
        sdir = os.path.join(SRC_STORAGE, pid)
        ddir = os.path.join(DEST_DEMO, "data", pid)
        os.makedirs(ddir, exist_ok=True)
        with open(os.path.join(sdir, "project.json"), encoding="utf-8") as f:
            proj = json.load(f)

        # floorplan
        fp = proj.get("floorplan") or {}
        if fp.get("url"):
            rel_fp = fp["url"].split(f"/files/{pid}/")[-1]
            copy_if_exists(os.path.join(sdir, rel_fp), os.path.join(ddir, rel_fp))

        # design_renders
        for sc in (proj.get("ai_design_schemes") or []):
            for key in ("render_rel_path", "ref_rel_path"):
                rp = sc.get(key)
                if rp:
                    copy_if_exists(os.path.join(sdir, rp), os.path.join(ddir, rp))

        # rooms & furniture
        for r in (proj.get("rooms") or []):
            rid = r["id"]
            if pid == "nest_test_v5":
                # 隐私脱敏：公开演示版不用私人卧室原图，改为干净的 3D 单房渲染图
                clean_shot_rel = f"docs/shots/clean_{rid}.jpg"
                fallback_shot_rel = "docs/shots/iso_clean.jpg"
                use_rel = clean_shot_rel if os.path.isfile(os.path.join(sdir, clean_shot_rel)) else fallback_shot_rel
                copy_if_exists(os.path.join(sdir, use_rel), os.path.join(ddir, use_rel))
                if r.get("photos"):
                    r["photos"] = [{
                        "id": f"ph_clean_{rid}",
                        "name": f"{r.get('name')}_3D实景视图.jpg",
                        "rel_path": use_rel,
                        "url": f"./data/{pid}/{use_rel}",
                    }]
            else:
                for ph in (r.get("photos") or []):
                    rel = ph.get("rel_path") or (ph.get("url") or "").split(f"/files/{pid}/")[-1]
                    if rel:
                        copy_if_exists(os.path.join(sdir, rel), os.path.join(ddir, rel))

            for it in ((r.get("scene_spec") or {}).get("furniture") or []):
                # GLB
                if it.get("glb_url"):
                    grel = it["glb_url"].split(f"/files/{pid}/")[-1]
                    copy_if_exists(os.path.join(sdir, grel), os.path.join(ddir, grel))
                # Clean-plate
                if it.get("clean_url"):
                    crel = it["clean_url"].split(f"/files/{pid}/")[-1]
                    copy_if_exists(os.path.join(sdir, crel), os.path.join(ddir, crel))
                # Crop: 对 nest_test_v5 优先用白底 clean_url 替代原裁图以彻底杜绝个人背景泄露
                if pid == "nest_test_v5" and it.get("clean_url"):
                    it["crop_url"] = it["clean_url"]
                elif it.get("crop_url"):
                    krel = it["crop_url"].split(f"/files/{pid}/")[-1]
                    copy_if_exists(os.path.join(sdir, krel), os.path.join(ddir, krel))

        # 去掉内部调试日志，减小公开 JSON 体积
        proj.pop("pipeline", None)
        with open(os.path.join(ddir, "project.json"), "w", encoding="utf-8") as f:
            json.dump(proj, f, ensure_ascii=False, indent=2)

        print(f"Synced {pid} -> {ddir}")

    print("Public demo sync complete:", DEST_DEMO)


if __name__ == "__main__":
    main()
