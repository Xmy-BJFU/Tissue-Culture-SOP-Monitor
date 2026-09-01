"""手部关键点验证：YOLO 手套框 + MediaPipe 21 点。

用来现场看三件事：
  1. 戴塑胶手套时点稳不稳、会不会整只手丢检
  2. 侧面/半侧面时会不会丢点
  3. 握瓶、握镊子、握美工刀时指尖会不会乱飘

不接 OCR / 追踪 / SOP。YOLO 只负责找出手套（以及瓶/镊/刀作对照）。

用法:
    python hand_mediapipe.py
    python hand_mediapipe.py --source 0
    python hand_mediapipe.py --source "视频.mp4"
    python hand_mediapipe.py --adapt try   # 手套外观适配，多方案试探

按键: Q 退出  S 保存  M 切换 ROI/整帧  C 切换肤色适配  空格暂停
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont
from ultralytics import YOLO

try:
    import mediapipe as mp
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, HandLandmarksConnections
    from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
except ImportError as exc:
    raise SystemExit(
        "未安装 mediapipe。请先在 yolov26 环境执行:\n"
        "    pip install mediapipe"
    ) from exc


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / r"runs\train\11n_100_deg45（26_17）\weights\best.pt"
# MediaPipe 原生库在 Windows 上打不开含中文的路径，模型放在纯英文目录
DEFAULT_MP_MODEL = Path(r"E:\XMY\weights\hand_landmarker.task")
HAND_CLASS = "手套"
CONTEXT_CLASSES = {"瓶子", "镊子", "美工刀"}
GLOVE_EXPAND = 1.55
MIN_CROP = 192
MAX_CROP = 512
# 不能微调 MediaPipe 权重；只能把手套裁图改成更像裸手再送进去。
# off=原图  lab=亮度保留改肤色  blue=蓝/紫丁腈  white=白手套  try=几种都试，谁出点用谁
ADAPT_MODES = ("off", "lab", "blue", "white", "try")
TIP_IDS = (4, 8, 12, 16, 20)  # 拇/食/中/无/小 指尖
STAT_WINDOW = 60
JITTER_WINDOW = 20

MP_MODEL_URLS = [
    "https://github.com/sanderdesnaijer/mediapipe-model-mirrors/releases/download/v1/hand_landmarker.task",
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
]

CHINESE_FONTS = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]

# BGR：掌、拇指、食指、中指、无名指、小指
PALM_COLOR = (240, 240, 240)
FINGER_COLORS = {
    1: (40, 140, 255),
    2: (40, 140, 255),
    3: (40, 140, 255),
    4: (40, 140, 255),
    5: (0, 220, 80),
    6: (0, 220, 80),
    7: (0, 220, 80),
    8: (0, 220, 80),
    9: (0, 220, 255),
    10: (0, 220, 255),
    11: (0, 220, 255),
    12: (0, 220, 255),
    13: (220, 80, 255),
    14: (220, 80, 255),
    15: (220, 80, 255),
    16: (220, 80, 255),
    17: (0, 255, 255),
    18: (0, 255, 255),
    19: (0, 255, 255),
    20: (0, 255, 255),
}
CONN_COLORS = {
    **{c.start: PALM_COLOR for c in HandLandmarksConnections.HAND_PALM_CONNECTIONS},
    **{c.start: FINGER_COLORS[4] for c in HandLandmarksConnections.HAND_THUMB_CONNECTIONS},
    **{c.start: FINGER_COLORS[8] for c in HandLandmarksConnections.HAND_INDEX_FINGER_CONNECTIONS},
    **{c.start: FINGER_COLORS[12] for c in HandLandmarksConnections.HAND_MIDDLE_FINGER_CONNECTIONS},
    **{c.start: FINGER_COLORS[16] for c in HandLandmarksConnections.HAND_RING_FINGER_CONNECTIONS},
    **{c.start: FINGER_COLORS[20] for c in HandLandmarksConnections.HAND_PINKY_FINGER_CONNECTIONS},
}


def load_font(size: int = 20) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in CHINESE_FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def put_cn(img: np.ndarray, lines: list[str], origin: tuple[int, int] = (12, 10)) -> np.ndarray:
    pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = load_font(20)
    x, y = origin
    for line in lines:
        bbox = draw.textbbox((x, y), line, font=font)
        draw.rectangle((bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2), fill=(0, 0, 0))
        color = (255, 220, 80) if "漏检" in line or "MISS" in line else (255, 255, 255)
        if "抖动" in line:
            color = (80, 255, 160)
        draw.text((x, y), line, font=font, fill=color)
        y = bbox[3] + 6
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _is_ascii_path(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def ensure_mp_model(path: Path) -> Path:
    """保证模型存在，并返回 MediaPipe 能打开的纯英文路径。"""
    ascii_path = path if _is_ascii_path(path) else DEFAULT_MP_MODEL
    local_copy = ROOT / "weights" / "hand_landmarker.task"
    for candidate in (path, ascii_path, local_copy):
        if candidate.exists() and candidate.stat().st_size > 1_000_000:
            if candidate != ascii_path:
                ascii_path.parent.mkdir(parents=True, exist_ok=True)
                if not ascii_path.exists() or ascii_path.stat().st_size != candidate.stat().st_size:
                    ascii_path.write_bytes(candidate.read_bytes())
            return ascii_path

    ascii_path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for url in MP_MODEL_URLS:
        try:
            print(f"正在下载 MediaPipe 手部模型:\n  {url}")
            urllib.request.urlretrieve(url, ascii_path)
            if ascii_path.exists() and ascii_path.stat().st_size > 1_000_000:
                print(f"已保存 {ascii_path}  ({ascii_path.stat().st_size / 1e6:.1f} MB)")
                return ascii_path
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"下载失败: {exc}")
    raise FileNotFoundError(
        f"拿不到 hand_landmarker.task。请手动下载后放到:\n  {ascii_path}\n最后错误: {last_err}"
    )


def open_source(source: str) -> tuple[cv2.VideoCapture, bool]:
    if source.isdigit():
        cam_id = int(source)
        cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            raise RuntimeError(f"打不开摄像头 {cam_id}。请关掉占用摄像头的软件后重试。")
        return cap, True
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"找不到视频: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {path}")
    return cap, False


def expand_xyxy(xyxy: np.ndarray, shape: tuple[int, ...], expand: float) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = max(x2 - x1, 8.0) * expand, max(y2 - y1, 8.0) * expand
    side = max(bw, bh)
    x1 = int(np.clip(cx - side / 2, 0, w - 1))
    y1 = int(np.clip(cy - side / 2, 0, h - 1))
    x2 = int(np.clip(cx + side / 2, 0, w - 1))
    y2 = int(np.clip(cy + side / 2, 0, h - 1))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return 0, 0, 0, 0
    return x1, y1, x2, y2


def _lab_to_skin(bgr: np.ndarray, target_a: float = 148.0, target_b: float = 142.0, mix: float = 0.78) -> np.ndarray:
    """保留亮度/皱褶，把色度拉向肤色。整块 ROI 都会变，背景也会偏肉色。"""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    A = A * (1.0 - mix) + target_a * mix
    B = B * (1.0 - mix) + target_b * mix
    out = cv2.merge([L, np.clip(A, 0, 255), np.clip(B, 0, 255)]).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


def _hsv_recolor(bgr: np.ndarray, hue: int, sat_lo: int, sat_hi: int) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    h, s, v = cv2.split(hsv)
    h[:] = hue
    s = np.clip(s + (sat_lo + sat_hi) // 2 - 40, sat_lo, sat_hi)
    out = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
    return cv2.cvtColor(out, cv2.COLOR_HSV2BGR)


def adapt_glove_crop(bgr: np.ndarray, mode: str) -> np.ndarray:
    """把手套裁图改成 MediaPipe 更熟的「裸手」外观。纹理尽量保留。"""
    if mode in ("off", "try") or bgr.size == 0:
        return bgr
    sharp = cv2.addWeighted(bgr, 1.25, cv2.GaussianBlur(bgr, (0, 0), 1.2), -0.25, 0)
    if mode == "lab":
        return _lab_to_skin(sharp)
    if mode == "blue":
        return _hsv_recolor(sharp, hue=16, sat_lo=50, sat_hi=140)
    if mode == "white":
        return _hsv_recolor(sharp, hue=18, sat_lo=40, sat_hi=110)
    return bgr


def mp_detect_crop(landmarker, crop: np.ndarray, mode: str):
    """返回 (result, 实际送进 MP 的图, 用上的适配名)。try 会依次试几种外观。"""
    order = ("lab", "blue", "white", "off") if mode == "try" else (mode,)
    last_img = crop
    last_res = None
    used = mode
    for name in order:
        img = adapt_glove_crop(crop, name)
        last_img = img
        last_res = landmarker.detect(bgr_to_mp_image(img))
        used = name
        if last_res.hand_landmarks:
            return last_res, img, used
    return last_res, last_img, used


def prepare_crop(bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """缩放到 MediaPipe 较稳的边长，返回 (图, 相对原裁切的缩放)。"""
    h, w = bgr.shape[:2]
    scale = 1.0
    short, long = min(h, w), max(h, w)
    if short < MIN_CROP:
        scale = MIN_CROP / max(short, 1)
    if long * scale > MAX_CROP:
        scale = MAX_CROP / long
    if abs(scale - 1.0) > 0.02:
        bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return bgr, scale


def bgr_to_mp_image(bgr: np.ndarray) -> mp.Image:
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def make_landmarker(model_path: Path, video: bool, num_hands: int, conf: float) -> HandLandmarker:
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionTaskRunningMode.VIDEO if video else VisionTaskRunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=conf,
        min_hand_presence_confidence=conf,
        min_tracking_confidence=max(conf - 0.1, 0.2),
    )
    return HandLandmarker.create_from_options(options)


def handedness_name(result, idx: int) -> str:
    if not result.handedness or idx >= len(result.handedness):
        return "?"
    cats = result.handedness[idx]
    if not cats:
        return "?"
    top = max(cats, key=lambda c: c.score or 0.0)
    name = (top.category_name or top.display_name or "?").lower()
    if "left" in name:
        return "左手"
    if "right" in name:
        return "右手"
    return top.category_name or "?"


def collect_hands(
    result,
    *,
    x0: float = 0.0,
    y0: float = 0.0,
    scale: float = 1.0,
    src_w: int,
    src_h: int,
) -> list[dict]:
    hands: list[dict] = []
    for i, lms in enumerate(result.hand_landmarks or []):
        pts = []
        for lm in lms:
            px = x0 + float(lm.x) * src_w / scale
            py = y0 + float(lm.y) * src_h / scale
            vis = lm.visibility if lm.visibility is not None else 1.0
            pts.append((px, py, vis))
        hands.append({"pts": pts, "name": handedness_name(result, i)})
    return hands


def draw_hand(img: np.ndarray, pts: list[tuple[float, float, float]], label: str) -> None:
    if len(pts) < 21:
        return
    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
        x1, y1, _ = pts[conn.start]
        x2, y2, _ = pts[conn.end]
        color = CONN_COLORS.get(conn.start, PALM_COLOR)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2, cv2.LINE_AA)
    for i, (x, y, vis) in enumerate(pts):
        r = 6 if i in TIP_IDS else 3
        color = FINGER_COLORS.get(i, PALM_COLOR)
        if vis is not None and vis < 0.5:
            cv2.circle(img, (int(x), int(y)), r, (80, 80, 80), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (int(x), int(y)), r, color, -1, cv2.LINE_AA)
            if i in TIP_IDS:
                cv2.circle(img, (int(x), int(y)), r + 2, (255, 255, 255), 1, cv2.LINE_AA)
    wx, wy, _ = pts[0]
    cv2.putText(img, label, (int(wx) + 8, int(wy) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)


def draw_obb(img: np.ndarray, pts: np.ndarray, color: tuple[int, int, int], text: str) -> None:
    poly = pts.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
    x, y = int(pts[:, 0].min()), int(max(pts[:, 1].min() - 6, 16))
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def parse_yolo(result) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
    """[(cls_name, conf, xyxy, obb_pts), ...]"""
    names = result.names
    out: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    if result.obb is not None and len(result.obb):
        pts_all = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        xyxy_all = result.obb.xyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        clss = result.obb.cls.cpu().numpy().astype(int)
        for pts, xyxy, conf, cls_id in zip(pts_all, xyxy_all, confs, clss):
            out.append((names.get(int(cls_id), str(cls_id)), float(conf), xyxy, pts.astype(np.float32)))
        return out
    if result.boxes is not None and len(result.boxes):
        xyxy_all = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for xyxy, conf, cls_id in zip(xyxy_all, confs, clss):
            x1, y1, x2, y2 = xyxy
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            out.append((names.get(int(cls_id), str(cls_id)), float(conf), xyxy, pts))
    return out


def tip_jitter_px(history: deque[np.ndarray]) -> float:
    if len(history) < 4:
        return 0.0
    arr = np.stack(list(history), axis=0)
    d = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    return float(np.mean(d))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO 手套框 + MediaPipe 21 点验证")
    p.add_argument("--source", default="0", help="摄像头编号或视频路径")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO-OBB 权重")
    p.add_argument("--mp-model", default=str(DEFAULT_MP_MODEL), help="hand_landmarker.task")
    p.add_argument("--mode", choices=("roi", "full"), default="roi", help="roi=只跑手套裁图；full=整帧")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度")
    p.add_argument("--mp-conf", type=float, default=0.25, help="MediaPipe 手检阈值，手套建议 0.2~0.3")
    p.add_argument(
        "--adapt",
        choices=ADAPT_MODES,
        default="try",
        help="手套外观适配: off原图 / lab肤色 / blue蓝紫丁腈 / white白手套 / try全试",
    )
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"找不到 YOLO 权重: {weights}")

    mp_model = ensure_mp_model(Path(args.mp_model))
    model = YOLO(str(weights))
    class_names = list(model.names.values())
    print(f"YOLO 类别: {class_names}")
    if HAND_CLASS not in class_names:
        print(f"警告: 模型没有「{HAND_CLASS}」，ROI 模式将找不到手套框，请改按 M 看整帧。")

    image_lm = make_landmarker(mp_model, video=False, num_hands=1, conf=args.mp_conf)
    video_lm = make_landmarker(mp_model, video=True, num_hands=2, conf=args.mp_conf)
    cap, is_cam = open_source(args.source)
    mode = args.mode
    adapt = args.adapt
    paused = False
    timestamp_ms = 0
    frame_id = 0
    t0 = time.time()
    fps_show = 0.0
    hit_hist: deque[int] = deque(maxlen=STAT_WINDOW)
    miss_hist: deque[int] = deque(maxlen=STAT_WINDOW)
    jitter_hist: deque[np.ndarray] = deque(maxlen=JITTER_WINDOW)
    snap_dir = ROOT / "runs" / "detect"
    snap_dir.mkdir(parents=True, exist_ok=True)

    print("窗口已开。请戴手套、侧面、握瓶/握刀各试一段。")
    print("看画面：绿骨架=检出；手套框红字 MISS=YOLO 有手套但 MediaPipe 没出点；食指白圈乱跳=指尖在飘。")
    print("C 切换外观适配（把手套改成更像裸手）。官方权重不能微调，这是唯一能改 MediaPipe 行为的办法。")
    print("按 Q 退出  S 保存  M 切换 ROI/整帧  C 切换适配  空格暂停")

    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    if is_cam:
                        print("读帧失败")
                        break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame_id += 1
                timestamp_ms += 33

            orig = frame
            yolo_out = model.predict(
                source=orig,
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                verbose=False,
            )[0]
            dets = parse_yolo(yolo_out)
            gloves = [d for d in dets if d[0] == HAND_CLASS]
            vis = orig.copy()

            for name, conf, _xyxy, pts in dets:
                if name == HAND_CLASS:
                    draw_obb(vis, pts, (0, 165, 255), f"{name} {conf:.2f}")
                elif name in CONTEXT_CLASSES:
                    draw_obb(vis, pts, (180, 180, 180), f"{name} {conf:.2f}")

            hands: list[dict] = []
            roi_preview = None
            used_adapt = adapt
            glove_miss = 0

            if mode == "full":
                mp_res = video_lm.detect_for_video(bgr_to_mp_image(orig), timestamp_ms)
                hands = collect_hands(mp_res, src_w=orig.shape[1], src_h=orig.shape[0])
                if gloves and not hands:
                    glove_miss = len(gloves)
            else:
                for _name, _conf, xyxy, _pts in gloves:
                    x1, y1, x2, y2 = expand_xyxy(xyxy, orig.shape, GLOVE_EXPAND)
                    if x2 <= x1 or y2 <= y1:
                        glove_miss += 1
                        continue
                    crop = orig[y1:y2, x1:x2]
                    proc, scale = prepare_crop(crop)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    mp_res, preview, used_adapt = mp_detect_crop(image_lm, proc, adapt)
                    if roi_preview is None:
                        roi_preview = preview
                    mapped = collect_hands(
                        mp_res,
                        x0=x1,
                        y0=y1,
                        scale=scale,
                        src_w=proc.shape[1],
                        src_h=proc.shape[0],
                    )
                    if mapped:
                        hands.extend(mapped)
                    else:
                        glove_miss += 1
                        cv2.putText(
                            vis,
                            "MISS",
                            (x1 + 6, y1 + 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            2,
                            cv2.LINE_AA,
                        )

            for hand in hands:
                draw_hand(vis, hand["pts"], hand["name"])

            n_tips = 0
            if hands:
                pts = hands[0]["pts"]
                n_tips = sum(1 for i in TIP_IDS if i < len(pts))
                ix, iy, _ = pts[8]
                jitter_hist.append(np.array([ix, iy], dtype=np.float32))
            else:
                jitter_hist.clear()

            hit_hist.append(1 if hands else 0)
            miss_hist.append(1 if glove_miss else 0)
            hit_rate = 100.0 * sum(hit_hist) / max(len(hit_hist), 1)
            miss_rate = 100.0 * sum(miss_hist) / max(len(miss_hist), 1)
            jitter = tip_jitter_px(jitter_hist)

            if frame_id % 10 == 0 and not paused:
                now = time.time()
                fps_show = 10.0 / max(now - t0, 1e-6)
                t0 = now

            if roi_preview is not None:
                thumb = cv2.resize(roi_preview, (160, 160))
                vis[8:168, vis.shape[1] - 168 : vis.shape[1] - 8] = thumb
                cv2.putText(
                    vis,
                    f"ROI {used_adapt}",
                    (vis.shape[1] - 160, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            hud = [
                f"模式 {'手套ROI' if mode == 'roi' else '整帧'}  适配 {adapt}   FPS {fps_show:.1f}",
                f"手套框 {len(gloves)}   MediaPipe手 {len(hands)}   指尖 {n_tips}/5",
                f"近{len(hit_hist)}帧检出率 {hit_rate:.0f}%   手套漏检 {miss_rate:.0f}%",
                f"食指抖动 {jitter:.1f} px   （握持静止时越大越飘）",
                "Q退出  S保存  M切换ROI  C切换适配  空格暂停",
            ]
            if glove_miss:
                hud.insert(2, f"本帧漏检 {glove_miss} 只手套（YOLO有框、关键点没有）")
            vis = put_cn(vis, hud)

            cv2.imshow("Hand MediaPipe", vis)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                snap = snap_dir / f"hand_mp_{frame_id:06d}.jpg"
                cv2.imwrite(str(snap), vis)
                print(f"已保存 {snap}")
            if key in (ord("m"), ord("M")):
                mode = "full" if mode == "roi" else "roi"
                jitter_hist.clear()
                print(f"切换到 {'整帧' if mode == 'full' else '手套ROI'}")
            if key in (ord("c"), ord("C")):
                adapt = ADAPT_MODES[(ADAPT_MODES.index(adapt) + 1) % len(ADAPT_MODES)]
                jitter_hist.clear()
                hit_hist.clear()
                miss_hist.clear()
                print(f"外观适配: {adapt}")
            if key == 32:
                paused = not paused
    finally:
        image_lm.close()
        video_lm.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
