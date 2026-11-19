#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GLB 四向证件照视觉定标器（PocketNest Stage 1）。

流程：
1. 读取 GLB 顶点计算 ±8° OBB 微去歪角（直角/轴对齐家具保持 0° 锐利，仅明显倾斜的矩形物体微调）。
2. 启动本地 HTTP 服务 + Playwright 无头 Chromium + 本地 Three.js/GLTFLoader（零外网 CDN 依赖），
   在纯浅色影棚背景 + 柔和多向照明 + 地面网格与 +Z 朝向箭头下，渲染 1024×1024 四宫格视图
   （YAW_0 / YAW_90 / YAW_180 / YAW_270）。
3. 将四宫格图 + 原家具裁图（若有）+ 家具类型/名称送入 gemini-4-hydrogen-high，
   按品类正面定义选出正面朝向镜头 (+Z) 的角度，得到：
     glb_fix_deg = (obb_micro_deg + chosen_yaw) % 360
4. 结果缓存至 <glb_path>.pose.json，避免重复调用；提供单件与批量共享浏览器接口。
"""

from __future__ import annotations

import base64
import functools
import http.server
import io
import json
import math
import os
import socketserver
import struct
import threading
import time
import urllib.parse
import uuid

from PIL import Image, ImageDraw, ImageFont

import pipeline_common as pc

CALIBRATOR_VERSION = "1.0-4view-vlm"

VENDOR_CANDIDATES = (
    "./frontend/vendor",
    "./frontend/vendor",
    "./frontend/vendor",
)

# 强轴对齐/直角品类：不进行 ±8° OBB 偏转，保持 90° 倍数锐利贴墙
AXIS_ALIGNED_TYPES = {
    "kitchen_counter",
    "wardrobe",
    "cabinet",
    "bookshelf",
    "fridge",
    "wall_cabinet",
    "vanity_sink",
    "toilet",
    "bathtub",
    "washer",
    "tv_stand",
    "tv",
    "bed",
    "desk",
    "sofa",
}

FRONT_DEFINITIONS = {
    "bed": "床 (bed)：【床尾】正对镜头与绿色 +Z 箭头，【床头靠背/枕头端】在远端后方（这样背贴墙 -Z 时，床头才能贴墙、床尾伸向房间）。",
    "sofa": "沙发 (sofa)：【坐垫前沿与敞开坐人面】正对镜头与绿色 +Z 箭头，【沙发高靠背】在远端后方。",
    "chair": "椅子 (chair)：【坐垫前沿与入座方向】正对镜头与绿色 +Z 箭头，【椅背】在远端后方。",
    "wardrobe": "衣柜 (wardrobe)：【对开/推拉柜门与把手面】正对镜头与绿色 +Z 箭头，【纯平板背面】在远端后方。",
    "cabinet": "储物柜 (cabinet)：【柜门/抽屉/开放格开口面】正对镜头与绿色 +Z 箭头，【背面背板】在远端后方。",
    "bookshelf": "书柜/置物架 (bookshelf)：【开放格/层板正面取物侧】正对镜头与绿色 +Z 箭头，【封闭背板或贴墙侧】在远端后方（转角置物架则以开口扇面朝向镜头）。",
    "fridge": "冰箱 (fridge)：【双门/单门与门把手正面】正对镜头与绿色 +Z 箭头，【背部散热面】在远端后方。",
    "kitchen_counter": "厨房地柜与台面 (kitchen_counter)：【地柜柜门/抽屉/操作正面】正对镜头与绿色 +Z 箭头（若为 L 型橱柜，主要柜门操作面朝向镜头与室内方向，贴墙封闭背板在后方）。",
    "wall_cabinet": "吊柜/壁挂置物架 (wall_cabinet)：【柜门/开放置物开口正面】正对镜头与绿色 +Z 箭头，【贴墙安装背面】在远端后方。",
    "vanity_sink": "洗手台 (vanity_sink)：【水盆开口/龙头前方/下方柜门正面】正对镜头与绿色 +Z 箭头，【贴墙背面】在远端后方。",
    "tv_stand": "电视柜/电视下置物架 (tv_stand)：【柜门/抽屉/开放格正面】正对镜头与绿色 +Z 箭头，【贴墙背面】在远端后方。",
    "washer": "洗衣机 (washer)：【圆形滚筒舱门与上方控制面板】正对镜头与绿色 +Z 箭头，【平板背面】在远端后方。",
    "desk": "书桌 (desk)：【放腿坐人侧/抽屉正面】正对镜头与绿色 +Z 箭头（若带桌面书架或挡板，书架/挡板在远端后方）。",
    "tv": "电视/显示器 (tv)：【黑色或发光屏幕正面】正对镜头与绿色 +Z 箭头，【支架或电源接口背面】在远端后方。",
    "toilet": "马桶 (toilet)：【马桶圈/坐便盆前缘】正对镜头与绿色 +Z 箭头，【水箱背面】在远端后方。",
    "bathtub": "浴缸 (bathtub)：【浴缸长边开口/进出使用侧】正对镜头与绿色 +Z 箭头。",
    "dining_table": "餐桌 (dining_table)：【长边正面】正对镜头与绿色 +Z 箭头（即长轴沿左右 X 方向横向展开；若是圆桌/正方桌，选结构最对称稳正的一格）。",
    "coffee_table": "茶几/边桌 (coffee_table)：【长边正面】正对镜头与绿色 +Z 箭头（若是圆桌/正方桌，选结构最对称稳正的一格）。",
    "side_table": "边桌 (side_table)：【长边正面或抽屉开口正面】正对镜头与绿色 +Z 箭头。",
}


def _find_vendor_dir() -> str:
    for d in VENDOR_CANDIDATES:
        if os.path.isfile(os.path.join(d, "three.module.js")) and os.path.isfile(
            os.path.join(d, "addons", "loaders", "GLTFLoader.js")
        ):
            return d
    raise RuntimeError(f"未找到本地 Three.js vendor 目录：{VENDOR_CANDIDATES}")


def _extract_glb_vertices(glb_path: str) -> list:
    with open(glb_path, "rb") as f:
        hdr = f.read(12)
        if len(hdr) < 12 or hdr[:4] != b"glTF":
            return []
        clen, _ = struct.unpack("<II", f.read(8))
        gjson = json.loads(f.read(clen).decode("utf-8"))
        bhdr = f.read(8)
        if len(bhdr) < 8:
            return []
        blen, _ = struct.unpack("<II", bhdr)
        buf = f.read(blen)

    pts = []
    for m in gjson.get("meshes", []):
        for p in m.get("primitives", []):
            pos_idx = (p.get("attributes") or {}).get("POSITION")
            if pos_idx is None:
                continue
            acc = gjson["accessors"][pos_idx]
            if acc.get("componentType") != 5126 or acc.get("type") != "VEC3":
                continue
            bv = gjson["bufferViews"][acc["bufferView"]]
            off = bv.get("byteOffset", 0) + acc.get("byteOffset", 0)
            st = bv.get("byteStride") or 12
            cnt = acc["count"]
            step = max(1, cnt // 3000)
            for i in range(0, cnt, step):
                if off + i * st + 12 <= len(buf):
                    pts.append(struct.unpack_from("<fff", buf, off + i * st))
    return pts


def compute_obb_micro_deg(glb_path: str, ftype: str) -> tuple[float, list[float]]:
    """计算 ±8° OBB 微去歪角度，并返回旋转后的基础 [sx, sy, sz]。

    轴对齐品类直接保持 0°；其余品类只有当倾斜角度 > 2° 且 XZ 包围盒面积缩减 >= 3%
    时才应用微纠偏，避免圆桌/柱状物体被误拧斜。
    """
    pts = _extract_glb_vertices(glb_path)
    if not pts:
        return 0.0, [0.8, 0.8, 0.6]

    xs0 = [p[0] for p in pts]
    ys0 = [p[1] for p in pts]
    zs0 = [p[2] for p in pts]
    sx0 = max(0.05, max(xs0) - min(xs0))
    sy0 = max(0.05, max(ys0) - min(ys0))
    sz0 = max(0.05, max(zs0) - min(zs0))
    area0 = sx0 * sz0

    best_deg = 0.0
    if ftype not in AXIS_ALIGNED_TYPES:
        cand_deg, best_area = 0.0, area0
        for deg in range(-8, 9):
            rad = math.radians(deg)
            c, s = math.cos(rad), math.sin(rad)
            xs = [p[0] * c + p[2] * s for p in pts]
            zs = [-p[0] * s + p[2] * c for p in pts]
            area = (max(xs) - min(xs)) * (max(zs) - min(zs))
            if area < best_area:
                best_area, cand_deg = area, float(deg)
        gain = (area0 - best_area) / max(area0, 1e-9)
        if abs(cand_deg) > 2.0 and gain >= 0.03:
            best_deg = cand_deg

    if best_deg != 0.0:
        rad = math.radians(best_deg)
        c, s = math.cos(rad), math.sin(rad)
        xs = [p[0] * c + p[2] * s for p in pts]
        zs = [-p[0] * s + p[2] * c for p in pts]
        sx0 = max(0.05, max(xs) - min(xs))
        sz0 = max(0.05, max(zs) - min(zs))

    return best_deg, [round(sx0, 3), round(sy0, 3), round(sz0, 3)]


def _viewer_html() -> str:
    return """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body { margin: 0; padding: 0; background: #f4f6f9; overflow: hidden; }
  canvas { display: block; width: 512px; height: 512px; }
</style>
<script type="importmap">
{
  "imports": {
    "three": "/vendor/three.module.js",
    "three/addons/": "/vendor/addons/"
  }
}
</script>
</head>
<body>
<script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const W = 512, H = 512;
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, alpha: false });
renderer.setSize(W, H, false);
renderer.setPixelRatio(1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf4f6f9);

const camera = new THREE.PerspectiveCamera(42, 1.0, 0.02, 200);

// 影棚级三点布光 + 柔光环境，确保黑白材质都能看清门缝、把手、屏幕、靠背
const hemi = new THREE.HemisphereLight(0xffffff, 0xd8dee9, 1.25);
scene.add(hemi);
const amb = new THREE.AmbientLight(0xffffff, 0.45);
scene.add(amb);

const keyLight = new THREE.DirectionalLight(0xfffcf7, 1.45);
keyLight.position.set(3.2, 5.5, 4.5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xeaf2ff, 0.85);
fillLight.position.set(-4.0, 3.0, 3.0);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffffff, 0.65);
rimLight.position.set(0.0, 3.0, -4.5);
scene.add(rimLight);

// 地面承影圆盘 + 网格 + 明确朝向镜头 (+Z) 的绿色前向箭头
const floorGeo = new THREE.PlaneGeometry(20, 20);
const floorMat = new THREE.MeshStandardMaterial({ color: 0xf4f6f9, roughness: 0.96, metalness: 0.0 });
const floor = new THREE.Mesh(floorGeo, floorMat);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.002;
floor.receiveShadow = true;
scene.add(floor);

const gridGroup = new THREE.Group();
scene.add(gridGroup);

const modelRoot = new THREE.Group();
scene.add(modelRoot);

const loader = new GLTFLoader();

function loadGlb(url) {
  return new Promise((resolve, reject) => {
    loader.load(url, (gltf) => resolve(gltf.scene), undefined, (err) => reject(err));
  });
}

window.renderGlbFourView = async function(glbUrl, obbMicroDeg) {
  while (modelRoot.children.length) modelRoot.remove(modelRoot.children[0]);
  while (gridGroup.children.length) gridGroup.remove(gridGroup.children[0]);

  const rawScene = await loadGlb(glbUrl);
  rawScene.traverse((n) => {
    if (n.isMesh) {
      n.castShadow = true;
      n.receiveShadow = true;
      if (n.material) {
        const mats = Array.isArray(n.material) ? n.material : [n.material];
        for (const m of mats) {
          if (m.roughness !== undefined && m.roughness < 0.15) m.roughness = 0.25;
        }
      }
    }
  });

  // 先用 0 yaw + obbMicroDeg 求基准尺寸
  const baseGroup = new THREE.Group();
  rawScene.rotation.y = (obbMicroDeg * Math.PI) / 180;
  baseGroup.add(rawScene);
  baseGroup.updateMatrixWorld(true);
  const box0 = new THREE.Box3().setFromObject(baseGroup);
  const sz0 = box0.getSize(new THREE.Vector3());
  const ctr0 = box0.getCenter(new THREE.Vector3());

  // 居中并贴地 y=0
  rawScene.position.set(-ctr0.x, -box0.min.y, -ctr0.z);
  baseGroup.updateMatrixWorld(true);

  const maxDim = Math.max(sz0.x, sz0.y, sz0.z, 0.1);
  const R = new THREE.Box3().setFromObject(baseGroup).getBoundingSphere(new THREE.Sphere()).radius || (maxDim * 0.65);

  // 动态适配地面网格与 +Z 前向箭头
  const gridSize = Math.max(1.2, R * 3.4);
  const grid = new THREE.GridHelper(gridSize, 12, 0xb8c4d4, 0xdbe2ec);
  grid.position.y = 0.001;
  gridGroup.add(grid);

  // 绿色实心前向箭头：从物体下方前端指向 +Z（相机方向）
  const arrowStart = new THREE.Vector3(0, 0.015, R * 0.55);
  const arrowDir = new THREE.Vector3(0, 0, 1);
  const arrowLen = Math.max(0.35, R * 0.85);
  const arrow = new THREE.ArrowHelper(arrowDir, arrowStart, arrowLen, 0x10b981, arrowLen * 0.36, arrowLen * 0.24);
  gridGroup.add(arrow);

  // 同样在两侧加淡淡的 Z 轴/X 轴参照线，防止旋转混淆
  const targetY = sz0.y * 0.46;
  const distMul = 2.05;
  camera.position.set(0, targetY + 0.75 * R * distMul, 1.55 * R * distMul);
  camera.lookAt(0, targetY, 0);
  camera.near = Math.max(0.01, R * 0.05);
  camera.far = R * 30;
  camera.updateProjectionMatrix();

  // 更新阴影相机范围
  const sSpan = R * 2.2;
  keyLight.shadow.camera.left = -sSpan;
  keyLight.shadow.camera.right = sSpan;
  keyLight.shadow.camera.top = sSpan;
  keyLight.shadow.camera.bottom = -sSpan;
  keyLight.shadow.camera.updateProjectionMatrix();

  modelRoot.add(baseGroup);

  const views = {};
  for (const yaw of [0, 90, 180, 270]) {
    // 绕 Y 轴旋转 yaw（因 baseGroup 已应用 obbMicroDeg 且以 XZ 原点居中）
    modelRoot.rotation.y = (yaw * Math.PI) / 180;
    modelRoot.updateMatrixWorld(true);
    renderer.render(scene, camera);
    views[String(yaw)] = renderer.domElement.toDataURL('image/png');
  }

  return {
    baseSize: [Number(sz0.x.toFixed(4)), Number(sz0.y.toFixed(4)), Number(sz0.z.toFixed(4))],
    views,
  };
};

window.__viewerReady = true;
</script>
</body>
</html>
"""


class _AssetHandler(http.server.BaseHTTPRequestHandler):
    vendor_dir: str = ""
    glb_registry: dict = {}

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path == "/viewer.html":
            body = _viewer_html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path.startswith("/vendor/"):
            rel = path[len("/vendor/"):]
            full = os.path.realpath(os.path.join(self.vendor_dir, rel))
            vroot = os.path.realpath(self.vendor_dir)
            if not full.startswith(vroot + os.sep) or not os.path.isfile(full):
                self.send_response(404)
                self.end_headers()
                return
            mime = "application/javascript; charset=utf-8" if full.endswith(".js") else "application/octet-stream"
            with open(full, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/glb/"):
            key = path[len("/glb/"):]
            fpath = self.glb_registry.get(key)
            if not fpath or not os.path.isfile(fpath):
                self.send_response(404)
                self.end_headers()
                return
            with open(fpath, "rb") as f:
                data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "model/gltf-binary")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_response(404)
        self.end_headers()


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class RenderSession:
    """可复用的无头 Chromium + 本地 Three.js 渲染会话。"""

    def __init__(self):
        self.vendor_dir = _find_vendor_dir()
        self.glb_registry: dict[str, str] = {}
        handler = type(
            "BoundAssetHandler",
            (_AssetHandler,),
            {"vendor_dir": self.vendor_dir, "glb_registry": self.glb_registry},
        )
        self.server = _ThreadedHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self._pw = None
        self._browser = None
        self._page = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--enable-unsafe-swiftshader",
                "--use-gl=swiftshader",
                "--js-flags=--max-old-space-size=4096",
            ]
        )
        ctx = self._browser.new_context(viewport={"width": 512, "height": 512})
        self._page = ctx.new_page()
        self._page.goto(f"http://127.0.0.1:{self.port}/viewer.html", wait_until="load", timeout=30000)
        self._page.wait_for_function("window.__viewerReady === true", timeout=20000)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._browser:
                self._browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                self._pw.stop()
        except Exception:
            pass
        try:
            self.server.shutdown()
            self.server.server_close()
        except Exception:
            pass

    def render_four_view(self, glb_path: str, obb_micro_deg: float, out_grid_path: str, label: str = "") -> list[float]:
        key = uuid.uuid4().hex + ".glb"
        self.glb_registry[key] = os.path.abspath(glb_path)
        try:
            res = self._page.evaluate(
                "([url, deg]) => window.renderGlbFourView(url, deg)",
                [f"/glb/{key}", float(obb_micro_deg)],
            )
        finally:
            self.glb_registry.pop(key, None)

        views = res["views"]
        base_size = [float(v) for v in res["baseSize"]]

        canvas = Image.new("RGB", (1024, 1024), (244, 246, 249))
        positions = {
            "0": (0, 0),
            "90": (512, 0),
            "180": (0, 512),
            "270": (512, 512),
        }
        badge_titles = {
            "0": "YAW_0 (0°)",
            "90": "YAW_90 (+90°)",
            "180": "YAW_180 (+180°)",
            "270": "YAW_270 (+270°)",
        }

        font = _load_font(22)
        small_font = _load_font(15)
        draw = ImageDraw.Draw(canvas)

        for yaw_str, (ox, oy) in positions.items():
            data_url = views[yaw_str]
            _, b64 = data_url.split(",", 1)
            tile = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            if tile.size != (512, 512):
                tile = tile.resize((512, 512), Image.Resampling.LANCZOS)
            canvas.paste(tile, (ox, oy))

            # 顶部状态标签
            title = badge_titles[yaw_str]
            tw = _text_w(draw, title, font)
            draw.rectangle([ox + 12, oy + 12, ox + 26 + tw, oy + 46], fill=(20, 30, 48))
            draw.text((ox + 19, oy + 17), title, fill=(255, 255, 255), font=font)

            # 右上角方向提示
            hint = "+Z -> CAMERA"
            hw = _text_w(draw, hint, small_font)
            draw.rectangle([ox + 512 - hw - 22, oy + 14, ox + 500, oy + 40], fill=(16, 185, 129))
            draw.text((ox + 512 - hw - 16, oy + 18), hint, fill=(255, 255, 255), font=small_font)

        # 十字分割线
        draw.line([(512, 0), (512, 1024)], fill=(148, 163, 184), width=3)
        draw.line([(0, 512), (1024, 512)], fill=(148, 163, 184), width=3)

        os.makedirs(os.path.dirname(os.path.abspath(out_grid_path)), exist_ok=True)
        canvas.save(out_grid_path, format="PNG")
        return base_size


def _load_font(size: int):
    for fpath in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(fpath):
            try:
                return ImageFont.truetype(fpath, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _text_w(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    try:
        return int(draw.textlength(text, font=font))
    except Exception:
        return len(text) * 12


def _front_rule_for(ftype: str, label: str) -> str:
    lbl = str(label or "").lower()
    if ("电视" in lbl or "显示器" in lbl or "tv" in lbl) and ftype == "other":
        return FRONT_DEFINITIONS["tv"]
    if ftype in FRONT_DEFINITIONS:
        return FRONT_DEFINITIONS[ftype]
    return (
        f"家具 ({label or ftype})：请让这件物体【常规使用时的正面/操作面/展示面】"
        "正对镜头与绿色 +Z 箭头，【贴墙背面或安装背板】在远端后方。"
    )


def _vlm_pick_yaw(
    grid_path: str,
    ftype: str,
    label: str,
    crop_path: str | None = None,
) -> dict:
    has_crop = bool(crop_path and os.path.isfile(crop_path))
    imgs = [grid_path] + ([crop_path] if has_crop else [])
    rule = _front_rule_for(ftype, label)
    crop_note = (
        "第 2 张图片是该家具的原始实景裁图，仅用于帮你辨认它原本长什么样、哪边是门/抽屉/正面。"
        if has_crop
        else "本次未提供原始实景裁图，请直接观察第 1 张 2×2 四宫格渲染图的几何与贴图细节。"
    )

    prompt = f"""你是 3D 家具模型的「证件照正面定标专家」。
第 1 张图是同一个 3D 家具模型绕竖直 Y 轴旋转 4 个角度后的 2×2 四宫格斜俯视图（camera 在 +Z 方向斜向下俯视原点，地面绿色箭头指向 +Z 相机方向）：
- 左上：`YAW_0` (0°)
- 右上：`YAW_90` (+90°)
- 左下：`YAW_180` (+180°)
- 右下：`YAW_270` (+270°)
{crop_note}

当前物体类别：`{ftype}`，名称：`{label or ftype}`。
【本品类标准正面定义（必须严格遵守）】
{rule}

【判别步骤】
1. 仔细对比 4 个格子里的正反面特征（如床头板 vs 床尾、椅背/沙发靠背 vs 坐垫前沿、柜门/把手/抽屉 vs 平板背面、冰箱门缝 vs 背面、洗衣机滚筒舱门 vs 背面、电视屏幕 vs 支架背板、马桶水箱 vs 马桶圈前缘、桌腿与书架朝向）。
2. 选出唯一一格，其中**标准正面正对着镜头（即朝向画面下方/绿色 +Z 箭头方向）**，且背面在远端后方。
3. 只能输出 0、90、180、270 四个整数之一。

严格输出纯 JSON：
{{
  "chosen_yaw": 0,
  "confidence": 0.95,
  "front_features_seen": "在选中格子里看到的正面关键特征",
  "reason": "简要说明为何选这一格而不是另外三格"
}}"""

    res = pc.call_vlm_json(
        prompt,
        imgs,
        timeout=140,
        max_images=2,
        model=pc.AUDIT_VLM_MODEL,
        max_tokens=16384,
    )
    yaw_raw = res.get("chosen_yaw")
    try:
        yaw = int(round(float(yaw_raw))) % 360
    except Exception as e:
        raise RuntimeError(f"VLM 返回非法 chosen_yaw={yaw_raw!r}: {res}") from e
    if yaw not in (0, 90, 180, 270):
        raise RuntimeError(f"VLM 返回非正交角度 chosen_yaw={yaw}: {res}")
    return {
        "chosen_yaw": yaw,
        "confidence": float(res.get("confidence") or 0.85),
        "front_features_seen": str(res.get("front_features_seen") or "")[:300],
        "reason": str(res.get("reason") or "")[:400],
        "vlm_model": (res.get("_vlm") or {}).get("model", pc.AUDIT_VLM_MODEL),
        "vlm_elapsed_s": (res.get("_vlm") or {}).get("elapsed_s"),
    }


def _pose_cache_path(glb_path: str) -> str:
    return f"{glb_path}.pose.json"


def _load_pose_cache(glb_path: str) -> dict | None:
    cpath = _pose_cache_path(glb_path)
    if not os.path.isfile(cpath):
        return None
    try:
        with open(cpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        st = os.stat(glb_path)
        if (
            data.get("version") == CALIBRATOR_VERSION
            and data.get("glb_size") == st.st_size
            and abs(float(data.get("glb_mtime", 0)) - st.st_mtime) < 1e-3
            and isinstance(data.get("glb_fix_deg"), (int, float))
            and isinstance(data.get("glb_aligned_size"), list)
            and len(data["glb_aligned_size"]) == 3
        ):
            return data
    except Exception:
        return None
    return None


def _save_pose_cache(glb_path: str, payload: dict):
    cpath = _pose_cache_path(glb_path)
    tmp = f"{cpath}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cpath)


def _rotated_size(base_size: list[float], yaw: int) -> list[float]:
    sx, sy, sz = [round(float(v), 3) for v in base_size]
    if yaw in (90, 270):
        return [sz, sy, sx]
    return [sx, sy, sz]


def calibrate_glb_pose(
    glb_path: str,
    ftype: str = "other",
    target_size: list | None = None,
    crop_path: str | None = None,
    label: str = "",
    debug_dir: str | None = None,
    force: bool = False,
    session: RenderSession | None = None,
) -> dict:
    """对单个 GLB 执行四向视觉定标并缓存 <glb_path>.pose.json。"""
    if not glb_path or not os.path.isfile(glb_path):
        raise FileNotFoundError(f"GLB 不存在: {glb_path}")

    if not force:
        cached = _load_pose_cache(glb_path)
        if cached:
            return cached

    obb_micro_deg, vert_base_size = compute_obb_micro_deg(glb_path, ftype)

    stem = os.path.splitext(os.path.basename(glb_path))[0]
    if debug_dir:
        grid_path = os.path.join(debug_dir, f"{stem}_4view.png")
    else:
        grid_path = os.path.join(os.path.dirname(glb_path), f"{stem}_4view.png")

    if session is not None:
        base_size = session.render_four_view(glb_path, obb_micro_deg, grid_path, label=label)
    else:
        with RenderSession() as sess:
            base_size = sess.render_four_view(glb_path, obb_micro_deg, grid_path, label=label)

    if not base_size or len(base_size) != 3:
        base_size = vert_base_size

    vlm_res = _vlm_pick_yaw(grid_path, ftype=ftype, label=label, crop_path=crop_path)
    chosen_yaw = int(vlm_res["chosen_yaw"])
    glb_fix_deg = round((obb_micro_deg + chosen_yaw) % 360.0, 2)
    aligned_size = _rotated_size(base_size, chosen_yaw)

    st = os.stat(glb_path)
    result = {
        "version": CALIBRATOR_VERSION,
        "glb_path": os.path.abspath(glb_path),
        "glb_size": st.st_size,
        "glb_mtime": st.st_mtime,
        "ftype": ftype,
        "label": label,
        "crop_path": os.path.abspath(crop_path) if crop_path and os.path.isfile(crop_path) else None,
        "grid_image_path": os.path.abspath(grid_path),
        "obb_micro_deg": obb_micro_deg,
        "chosen_yaw": chosen_yaw,
        "glb_fix_deg": glb_fix_deg,
        "glb_aligned_size": aligned_size,
        "pose_method": "vlm_4view_visual",
        "confidence": vlm_res["confidence"],
        "front_features_seen": vlm_res["front_features_seen"],
        "reason": vlm_res["reason"],
        "vlm_model": vlm_res["vlm_model"],
        "vlm_elapsed_s": vlm_res["vlm_elapsed_s"],
        "calibrated_at": int(time.time()),
    }
    _save_pose_cache(glb_path, result)
    return result


def calibrate_batch(
    items: list[dict],
    debug_dir: str,
    force: bool = False,
    max_vlm_workers: int = 4,
) -> list[dict]:
    """批量定标：单浏览器复用渲染全部 4view 图，再并发调用 VLM 定标并落盘 .pose.json。"""
    from concurrent.futures import ThreadPoolExecutor

    os.makedirs(debug_dir, exist_ok=True)
    results: list[dict | None] = [None] * len(items)
    pending: list[tuple[int, dict, float, list[float], str]] = []

    for idx, it in enumerate(items):
        glb_path = it["glb_path"]
        if not force:
            cached = _load_pose_cache(glb_path)
            if cached:
                results[idx] = cached
                continue
        obb_deg, vert_sz = compute_obb_micro_deg(glb_path, it.get("ftype", "other"))
        stem = os.path.splitext(os.path.basename(glb_path))[0]
        grid_path = os.path.join(debug_dir, f"{stem}_4view.png")
        pending.append((idx, it, obb_deg, vert_sz, grid_path))

    if not pending:
        return [r for r in results if r is not None]

    rendered: list[tuple[int, dict, float, list[float], str]] = []
    with RenderSession() as sess:
        for idx, it, obb_deg, vert_sz, grid_path in pending:
            base_sz = sess.render_four_view(
                it["glb_path"], obb_deg, grid_path, label=it.get("label", "")
            )
            rendered.append((idx, it, obb_deg, base_sz or vert_sz, grid_path))

    def _finish_one(job):
        idx, it, obb_deg, base_sz, grid_path = job
        glb_path = it["glb_path"]
        ftype = it.get("ftype", "other")
        label = it.get("label", "")
        crop_path = it.get("crop_path")
        vlm_res = _vlm_pick_yaw(grid_path, ftype=ftype, label=label, crop_path=crop_path)
        chosen_yaw = int(vlm_res["chosen_yaw"])
        glb_fix_deg = round((obb_deg + chosen_yaw) % 360.0, 2)
        aligned_size = _rotated_size(base_sz, chosen_yaw)
        st = os.stat(glb_path)
        res = {
            "version": CALIBRATOR_VERSION,
            "glb_path": os.path.abspath(glb_path),
            "glb_size": st.st_size,
            "glb_mtime": st.st_mtime,
            "ftype": ftype,
            "label": label,
            "crop_path": os.path.abspath(crop_path) if crop_path and os.path.isfile(crop_path) else None,
            "grid_image_path": os.path.abspath(grid_path),
            "obb_micro_deg": obb_deg,
            "chosen_yaw": chosen_yaw,
            "glb_fix_deg": glb_fix_deg,
            "glb_aligned_size": aligned_size,
            "pose_method": "vlm_4view_visual",
            "confidence": vlm_res["confidence"],
            "front_features_seen": vlm_res["front_features_seen"],
            "reason": vlm_res["reason"],
            "vlm_model": vlm_res["vlm_model"],
            "vlm_elapsed_s": vlm_res["vlm_elapsed_s"],
            "calibrated_at": int(time.time()),
        }
        _save_pose_cache(glb_path, res)
        return idx, res

    with ThreadPoolExecutor(max_workers=max(1, min(max_vlm_workers, len(rendered)))) as ex:
        for idx, res in ex.map(_finish_one, rendered):
            results[idx] = res

    return [r for r in results if r is not None]


def _items_from_project(project_json_path: str, models_dir: str | None = None) -> list[dict]:
    pdir = os.path.dirname(os.path.abspath(project_json_path))
    with open(project_json_path, "r", encoding="utf-8") as f:
        proj = json.load(f)
    meta_by_id = {}
    for room in proj.get("rooms", []):
        for it in (room.get("scene_spec") or {}).get("furniture", []):
            iid = it.get("id")
            if iid:
                crop_rel = it.get("crop_rel")
                crop_abs = os.path.join(pdir, crop_rel) if crop_rel else None
                meta_by_id[iid] = {
                    "ftype": it.get("type", "other"),
                    "label": it.get("label", ""),
                    "crop_path": crop_abs if (crop_abs and os.path.isfile(crop_abs)) else None,
                    "room": room.get("name", ""),
                }

    scan_dir = models_dir or os.path.join(pdir, "models_opt")
    items = []
    for fname in sorted(os.listdir(scan_dir)):
        if not fname.lower().endswith(".glb"):
            continue
        stem = os.path.splitext(fname)[0]
        meta = meta_by_id.get(stem, {})
        items.append({
            "id": stem,
            "glb_path": os.path.join(scan_dir, fname),
            "ftype": meta.get("ftype", "other"),
            "label": meta.get("label", stem),
            "crop_path": meta.get("crop_path"),
            "room": meta.get("room", ""),
        })
    return items


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GLB 四向证件照视觉定标器")
    parser.add_argument("glb_path", nargs="?", help="单个 GLB 路径")
    parser.add_argument("--project", help="project.json 路径（批量扫描 models_opt）")
    parser.add_argument("--models-dir", help="指定 GLB 目录")
    parser.add_argument("--debug-dir", help="四宫格输出目录")
    parser.add_argument("--ftype", default="other", help="单件类别")
    parser.add_argument("--label", default="", help="单件名称")
    parser.add_argument("--crop", default="", help="原裁图路径")
    parser.add_argument("--force", action="store_true", help="忽略 .pose.json 缓存强制重算")
    parser.add_argument("--workers", type=int, default=4, help="VLM 并发线程数")
    args = parser.parse_args()

    if args.project:
        pdir = os.path.dirname(os.path.abspath(args.project))
        debug_dir = args.debug_dir or os.path.join(pdir, "debug", "glb_4view")
        jobs = _items_from_project(args.project, models_dir=args.models_dir)
        out = calibrate_batch(jobs, debug_dir=debug_dir, force=args.force, max_vlm_workers=args.workers)
        for r in out:
            print(
                f"{os.path.basename(r['glb_path']):20s} | {r['ftype']:16s} | {r['label']:14s} | "
                f"obb={r['obb_micro_deg']:+5.1f} yaw={r['chosen_yaw']:3d} fix={r['glb_fix_deg']:6.1f} | "
                f"sz={r['glb_aligned_size']} | {r['front_features_seen'][:48]}"
            )
    elif args.glb_path:
        res = calibrate_glb_pose(
            args.glb_path,
            ftype=args.ftype,
            crop_path=args.crop or None,
            label=args.label,
            debug_dir=args.debug_dir,
            force=args.force,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))
    else:
        parser.print_help()
