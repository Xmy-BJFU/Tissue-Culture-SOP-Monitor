"""实时监测：YOLO-OBB + ByteTrack + 瓶子 OCR + 手套 RTMPose + 动作规则。.

本文件独立运行，不依赖项目里其它 .py。
画面上不叠检出率、漏检、抖动等统计字。

动作（路线 A，几何规则，不训练行为模型）：
    手持、倾倒开始、倒完回正（计时零点）、倒出、灭菌、摇晃、切割、斜插
切割：手持美工刀 + 近 15 帧刀心移动跨度 ≥ 55px，连续 12 帧。
斜插仍依赖后续「种球、鳞茎、培养基」检测类，类别未到齐时只显示等待。

用法:
    python video_track_ocr_hand_pose.py
    python video_track_ocr_hand_pose.py --source 0
    python video_track_ocr_hand_pose.py --hand-mode glove
    python video_track_ocr_hand_pose.py --hand-mode bare

按键: Q 退出  S 保存  H 切换戴手套/不戴手套  R 重置动作  空格暂停
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ocr_track_bind import (
    crowded_bottle_tids,
    merge_ocr_cache,
    neighbor_crop_expand,
    neighbor_label_conflict,
    ocr_blocked_tids,
    rebind_ocr_cache,
    should_run_ocr,
)

try:
    from rtmlib import RTMDet, RTMPose
except ImportError as exc:
    raise SystemExit(
        "未安装 rtmlib。请先在 yolov26 环境执行:\n"
        "    pip install rtmlib onnxruntime-gpu\n"
        "若没有 CUDA，改成: pip install rtmlib onnxruntime"
    ) from exc

DEFAULT_WEIGHTS = ROOT / r"runs\train\11n_100_deg45（26_17）\weights\best.pt"
HAND_CLASS = "手套"
OCR_CLASS = "瓶子"
GLOVE_EXPAND = 1.55
MIN_VALID_KPTS = 8
TIP_IDS = (4, 8, 12, 16, 20)

OCR_EVERY = 4
OCR_RETRY = 6
OCR_REFRESH = 18
MOVE_THRESH = 40.0
CROP_EXPAND = 1.12
MIN_CROP_SIDE = 48
MAX_CROP_SIDE = 960
DET_THRESH = 0.3
DET_BOX_THRESH = 0.6
DET_UNCLIP = 1.5
REC_SCORE_THRESH = 0.5
TRACK_BUFFER = 150
TRACK_MATCH_THRESH = 0.85
OCR_HOLD_FRAMES = 180
OCR_INHERIT_DIST = 140.0
OCR_INHERIT_GAP = 48.0
OCR_STICKY_DIST = 52.0

RTMDET_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
)
RTMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
)

REAGENT_LABELS = ("酒精", "无菌水", "次氯酸钠", "灭菌瓶", "培养基")
REAGENT_ALIASES = {
    "酒精": ("酒精", "乙醇", "alcohol", "etoh"),
    "无菌水": ("无菌水", "无菌", "灭菌水", "蒸馏水"),
    "次氯酸钠": ("次氯酸钠", "次氯酸", "次氯", "84"),
    "灭菌瓶": ("灭菌瓶", "灭菌罐", "灭菌"),
    "培养基": ("培养基", "培养皿"),
}
REAGENT_CHAR_WEIGHTS = {
    "酒精": {"酒": 3.0, "精": 3.0, "乙": 2.0, "醇": 2.0},
    "无菌水": {"无": 3.0, "菌": 3.0, "水": 1.5, "天": 0.8},
    "次氯酸钠": {"氯": 3.0, "钠": 3.0, "次": 2.0, "酸": 1.5},
    "灭菌瓶": {"灭": 3.0, "瓶": 2.5, "菌": 1.0},
    "培养基": {"培": 3.0, "养": 3.0, "基": 2.5},
}

ROLE_REAGENTS = ("酒精", "无菌水", "次氯酸钠")
ROLE_STERILE_BOTTLE = "灭菌瓶"
ROLE_MEDIUM = "培养基"
CLS_TWEEZER = "镊子"
CLS_KNIFE = "美工刀"
CLS_STERILIZER = "灭菌器"
BULB_CLASSES = ("种球", "鳞茎")
MEDIUM_CLASSES = ("培养基",)
HOLDABLE_CLASSES = {"瓶子", "镊子", "美工刀", "种球", "鳞茎", "培养基"}
SKIP_HOLD_CLASSES = {"手套", "灭菌器", "口罩", "帽子"}

HOLD_ON_FRAMES = 5
HOLD_OFF_FRAMES = 6
HOLD_DIST_PX = 16.0
HOLD_DIST_SCALE = 0.28
HOLD_MIN_HITS = 2
HOLD_SHRINK = 0.78
HOLD_INSIDE_ONE = 6.0
POUR_TILT_DEG = 60.0
POUR_UPRIGHT_DEG = 14.0
POUR_ON_FRAMES = 5
POUR_OFF_FRAMES = 6
POUR_NEAR_SCALE = 1.8
SHAKE_WINDOW = 20
SHAKE_ANGLE_STD = 5.5
SHAKE_MIN_RANGE = 12.0
SHAKE_MAX_SHIFT = 48.0
SHAKE_ON_FRAMES = 8
HOLE_TOP_FRAC = 0.32
HOLE_SIDE_SHRINK = 0.18
IN_HOLE_FRAMES = 6
OPENING_FRAMES = 90
CUT_WINDOW = 15
CUT_SPAN_PX = 55.0
CUT_HOLD_FRAMES = 12
INSERT_IOU = 0.02
INSERT_ANGLE_OK = (25.0, 55.0)
HIST_LEN = 36
RECENT_HOLD_FRAMES = 18
SOAK_SEC = {"酒精": 30.0, "无菌水": 30.0, "次氯酸钠": 15 * 60.0}
SOAK_MARGIN = 0.10
SOAK_DUMP_TILT = 60.0
SOAK_DUMP_FRAMES = 8
SOAK_DUMP_MAX_REVERSALS = 1

# MediaPipe 同款 21 点连线，避免再依赖 hand_mediapipe.py
HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)
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


@dataclass
class Detection:
    pts: np.ndarray
    xyxy: np.ndarray
    xywh: np.ndarray
    conf: float
    cls_id: int


@dataclass
class OcrRecord:
    raw_text: str = ""
    score: float = 0.0
    reagent: str = ""
    center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    size: float = 0.0
    height: float = 0.0
    frame_id: int = 0
    ocr_frame: int = 0


class AABBDetections:
    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xywh = np.asarray(xywh, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return int(self.conf.shape[0])

    def __getitem__(self, item) -> AABBDetections:
        return AABBDetections(self.xywh[item], self.conf[item], self.cls[item])


def match_reagent_label(ocr_text: str) -> str:
    text = "".join((ocr_text or "").split())
    if not text:
        return ""
    hay = text.casefold()
    alias_hits = []
    for label, aliases in REAGENT_ALIASES.items():
        for alias in (label, *aliases):
            if alias and alias.casefold() in hay:
                alias_hits.append((len(alias), label))
    if alias_hits:
        alias_hits.sort(key=lambda x: x[0], reverse=True)
        return alias_hits[0][1]
    scored = []
    for label in REAGENT_LABELS:
        weights = REAGENT_CHAR_WEIGHTS[label]
        score = sum(w for ch, w in weights.items() if ch in text)
        consecutive = sum(2.0 for i in range(len(label) - 1) if label[i : i + 2] in text)
        n_hit = sum(1 for ch in weights if ch in text)
        if (score > 0 or consecutive > 0) and (n_hit >= 2 or consecutive > 0):
            scored.append((score + consecutive, consecutive, label))
    if not scored:
        return ""
    scored.sort(reverse=True)
    return scored[0][2]


def bottle_label(ocr_text: str) -> str:
    name = match_reagent_label(ocr_text)
    if not name:
        return OCR_CLASS
    if name in ("灭菌瓶", "培养基"):
        return name
    return f"{name}瓶"


def _get_field(item, key: str):
    if item is None:
        return None
    if isinstance(item, dict):
        return item.get(key)
    try:
        return item[key]
    except Exception:
        pass
    if hasattr(item, "get"):
        try:
            return item.get(key)
        except Exception:
            pass
    return getattr(item, key, None)


def _upright_obb_pts(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    nxt = np.roll(pts, -1, axis=0)
    area = float(np.sum(pts[:, 0] * nxt[:, 1] - nxt[:, 0] * pts[:, 1]))
    if area < 0:
        pts = pts[::-1].copy()
    mids = (pts + np.roll(pts, -1, axis=0)) / 2.0
    pts = np.roll(pts, -int(np.argmin(mids[:, 1])), axis=0)
    return pts


def crop_obb(image: np.ndarray, pts: np.ndarray, expand: float = 1.08) -> np.ndarray | None:
    pts = _upright_obb_pts(pts)
    center = pts.mean(axis=0, keepdims=True)
    pts = center + (pts - center) * expand
    w = int(max(np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])))
    h = int(max(np.linalg.norm(pts[1] - pts[2]), np.linalg.norm(pts[3] - pts[0])))
    w, h = max(w, 8), max(h, 8)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(pts, dst), (w, h))
    ch, cw = crop.shape[:2]
    if min(ch, cw) < 64:
        scale = 64.0 / max(min(ch, cw), 1)
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def crop_aabb(image: np.ndarray, xyxy: np.ndarray, expand: float = 1.08) -> np.ndarray | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = xyxy.tolist()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = (x2 - x1) * expand, (y2 - y1) * expand
    x1 = int(np.clip(cx - bw / 2, 0, w - 1))
    y1 = int(np.clip(cy - bh / 2, 0, h - 1))
    x2 = int(np.clip(cx + bw / 2, 0, w - 1))
    y2 = int(np.clip(cy + bh / 2, 0, h - 1))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return image[y1:y2, x1:x2]


def prepare_crop(crop: np.ndarray) -> np.ndarray:
    h, w = crop.shape[:2]
    scale = 1.0
    if min(h, w) < MIN_CROP_SIDE:
        scale = MIN_CROP_SIDE / max(min(h, w), 1)
    if max(h, w) * scale > MAX_CROP_SIDE:
        scale = MAX_CROP_SIDE / max(h, w)
    if abs(scale - 1.0) > 0.05:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def load_ocr():
    from paddleocr import PaddleOCR

    print("正在加载 PP-OCRv6_tiny ...")
    ocr = PaddleOCR(
        device="cpu",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv6_tiny_det",
        text_recognition_model_name="PP-OCRv6_tiny_rec",
    )
    print("PaddleOCR 就绪（PP-OCRv6 tiny）")
    return ocr


def run_ocr(ocr, crop: np.ndarray) -> tuple[str, float]:
    if crop is None or crop.size == 0:
        return "", 0.0
    result = ocr.predict(
        prepare_crop(crop),
        text_det_thresh=DET_THRESH,
        text_det_box_thresh=DET_BOX_THRESH,
        text_det_unclip_ratio=DET_UNCLIP,
        text_rec_score_thresh=REC_SCORE_THRESH,
    )
    texts, scores = [], []
    for item in result or []:
        rec_texts = _get_field(item, "rec_texts") or _get_field(item, "rec_text") or []
        rec_scores = _get_field(item, "rec_scores") or _get_field(item, "rec_score") or []
        if isinstance(rec_texts, str):
            rec_texts = [rec_texts]
            rec_scores = [rec_scores] if not isinstance(rec_scores, (list, tuple)) else rec_scores
        for i, text in enumerate(rec_texts):
            score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
            if text and score >= REC_SCORE_THRESH:
                texts.append(str(text).strip())
                scores.append(score)
    if not texts:
        return "", 0.0
    return " ".join(texts), float(np.mean(scores))


def load_bytetrack(track_buffer: int) -> BYTETracker:
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    cfg.track_buffer = track_buffer
    cfg.match_thresh = TRACK_MATCH_THRESH
    return BYTETracker(args=cfg)


def parse_detections(result) -> list[Detection]:
    dets: list[Detection] = []
    if result.obb is not None and len(result.obb):
        pts_all = result.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        xyxy_all = result.obb.xyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        clss = result.obb.cls.cpu().numpy().astype(int)
        for pts, xyxy, conf, cls_id in zip(pts_all, xyxy_all, confs, clss):
            dets.append(_make_det(pts, xyxy, float(conf), int(cls_id)))
        return dets
    if result.boxes is not None and len(result.boxes):
        xyxy_all = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for xyxy, conf, cls_id in zip(xyxy_all, confs, clss):
            x1, y1, x2, y2 = xyxy
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            dets.append(_make_det(pts, xyxy, float(conf), int(cls_id)))
    return dets


def _make_det(pts: np.ndarray, xyxy: np.ndarray, conf: float, cls_id: int) -> Detection:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    xywh = np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, 1.0), max(y2 - y1, 1.0)], dtype=np.float32)
    return Detection(
        pts=pts.astype(np.float32).reshape(4, 2),
        xyxy=np.array([x1, y1, x2, y2], dtype=np.float32),
        xywh=xywh,
        conf=conf,
        cls_id=cls_id,
    )


def detections_to_aabb(dets: list[Detection]) -> AABBDetections:
    if not dets:
        return AABBDetections(np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,)))
    return AABBDetections(
        np.stack([d.xywh for d in dets], axis=0),
        np.array([d.conf for d in dets], dtype=np.float32),
        np.array([d.cls_id for d in dets], dtype=np.float32),
    )


def update_tracks(
    tracker: BYTETracker,
    dets: list[Detection],
    frame: np.ndarray,
) -> tuple[list[tuple[int, Detection, float]], list[int]]:
    tracks = tracker.update(detections_to_aabb(dets), frame)
    removed = [int(t.track_id) for t in getattr(tracker, "removed_stracks_frame", [])]
    active: list[tuple[int, Detection, float]] = []
    if tracks is None or len(tracks) == 0:
        return active, removed
    for row in np.atleast_2d(tracks):
        tid = int(row[4])
        idx = int(row[7])
        score = float(row[5])
        if 0 <= idx < len(dets):
            active.append((tid, dets[idx], score))
    return active, removed


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


def draw_hand(img: np.ndarray, pts: list[tuple[float, float, float]]) -> None:
    if len(pts) < 21:
        return
    for a, b in HAND_CONNECTIONS:
        x1, y1, _ = pts[a]
        x2, y2, _ = pts[b]
        color = FINGER_COLORS.get(a, PALM_COLOR)
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


def yolo_device_to_rtm(device: str) -> str:
    d = str(device).lower()
    if d in {"cpu", "-1"}:
        return "cpu"
    if d.startswith("cuda"):
        return d if ":" in d else "cuda"
    return "cuda"


def enable_ort_cuda() -> None:
    try:
        import torch

        lib = Path(torch.__file__).resolve().parent / "lib"
        if not lib.is_dir():
            return
        os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib))
    except Exception:
        pass


def make_rtm(device: str, det_thr: float) -> tuple[RTMDet, RTMPose]:
    if device != "cpu":
        enable_ort_cuda()
    print("加载 RTMPose 预训练（首次会下载 ONNX，请稍等）...")
    try:
        return _build_rtm(device, det_thr)
    except Exception as exc:
        if device == "cpu":
            raise
        print(f"CUDA 加载失败，改用 CPU: {exc}")
        return _build_rtm("cpu", det_thr)


def _build_rtm(device: str, det_thr: float) -> tuple[RTMDet, RTMPose]:
    det = RTMDet(
        RTMDET_URL,
        model_input_size=(320, 320),
        backend="onnxruntime",
        device=device,
        score_thr=det_thr,
    )
    pose = RTMPose(
        RTMPOSE_URL,
        model_input_size=(256, 256),
        to_openpose=False,
        backend="onnxruntime",
        device=device,
    )
    providers = list(pose.session.get_providers())
    print(f"RTMPose 设备 {device}  providers {providers}")
    return det, pose


def boxes_to_list(bboxes) -> list[list[float]]:
    if bboxes is None:
        return []
    arr = np.asarray(bboxes)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = []
    for row in arr:
        x1, y1, x2, y2 = [float(v) for v in row[:4]]
        if x2 - x1 >= 8 and y2 - y1 >= 8:
            out.append([x1, y1, x2, y2])
    return out


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def merge_boxes(primary: list[list[float]], extra: list[list[float]], iou_thr: float = 0.45) -> list[list[float]]:
    out = [list(b) for b in primary]
    for box in extra:
        if all(iou_xyxy(box, p) < iou_thr for p in out):
            out.append(list(box))
    return out


def glove_boxes_from_yolo(gloves, shape) -> list[list[float]]:
    boxes: list[list[float]] = []
    for _name, _conf, xyxy, _pts in gloves:
        x1, y1, x2, y2 = expand_xyxy(xyxy, shape, GLOVE_EXPAND)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
    return boxes


def pose_on_boxes(pose: RTMPose, bgr: np.ndarray, bboxes: list[list[float]], pose_conf: float) -> list[dict]:
    if not bboxes:
        return []
    kpts, scores = pose(bgr, bboxes=bboxes)
    hands: list[dict] = []
    kpts = np.asarray(kpts)
    scores = np.asarray(scores)
    if kpts.ndim == 2:
        kpts = kpts[None, ...]
        scores = scores[None, ...]
    for xy, sc in zip(kpts, scores):
        if xy.shape[0] < 21:
            continue
        sc = np.asarray(sc, dtype=np.float32).reshape(-1)
        n_ok = int((sc >= pose_conf).sum())
        mean_sc = float(sc.mean()) if sc.size else 0.0
        if n_ok < MIN_VALID_KPTS or mean_sc < pose_conf:
            continue
        pts = [(float(x), float(y), float(v)) for (x, y), v in zip(xy, sc)]
        hands.append({"pts": pts, "score": mean_sc})
    hands.sort(key=lambda h: h["score"], reverse=True)
    return hands


def obb_tilt_from_vertical(pts: np.ndarray) -> float:
    """长轴相对画面竖直方向的倾角，0=直立，90=横躺。."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    axis = e01 if float(np.linalg.norm(e01)) >= float(np.linalg.norm(e12)) else e12
    ang = abs(np.degrees(np.arctan2(float(axis[0]), float(axis[1])))) % 180.0
    if ang > 90.0:
        ang = 180.0 - ang
    return float(ang)


def obb_axis(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    e01 = pts[1] - pts[0]
    e12 = pts[2] - pts[1]
    axis = e01 if float(np.linalg.norm(e01)) >= float(np.linalg.norm(e12)) else e12
    n = float(np.linalg.norm(axis))
    if n < 1e-6:
        return np.array([0.0, 1.0], dtype=np.float32)
    return (axis / n).astype(np.float32)


def axes_included_deg(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    a, b = obb_axis(pts_a), obb_axis(pts_b)
    cos = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def point_in_xyxy(x: float, y: float, xyxy: np.ndarray) -> bool:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return x1 <= x <= x2 and y1 <= y <= y2


def xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    den = area_a + area_b - inter
    return inter / den if den > 0 else 0.0


def hole_xyxy_from_sterilizer(xyxy: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    return np.array(
        [
            x1 + w * HOLE_SIDE_SHRINK,
            y1,
            x2 - w * HOLE_SIDE_SHRINK,
            y1 + h * HOLE_TOP_FRAC,
        ],
        dtype=np.float32,
    )


def in_hole(item: TrackedItem, hole: np.ndarray) -> bool:
    cx, cy = float(item.center[0]), float(item.center[1])
    if point_in_xyxy(cx, cy, hole):
        return True
    if xyxy_iou(item.xyxy, hole) >= 0.05:
        return True
    for x, y in item.pts:
        if point_in_xyxy(float(x), float(y), hole):
            return True
    return False


def hand_probe_points(hand: dict) -> list[tuple[float, float]]:
    pts = hand.get("pts") or []
    if len(pts) < 21:
        return []
    out: list[tuple[float, float]] = []
    for i in (0, 4, 8, 12, 16, 20, 5, 9):
        x, y, vis = pts[i]
        if vis is None or vis >= 0.45:
            out.append((float(x), float(y)))
    palm = [pts[i] for i in (0, 5, 9, 13, 17) if pts[i][2] is None or pts[i][2] >= 0.35]
    if palm:
        out.append((float(np.mean([p[0] for p in palm])), float(np.mean([p[1] for p in palm]))))
    return out


def shrink_obb(pts: np.ndarray, scale: float) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    c = pts.mean(axis=0, keepdims=True)
    return c + (pts - c) * float(scale)


def hold_threshold(item: TrackedItem) -> float:
    w = float(item.det.xywh[2])
    h = float(item.det.xywh[3])
    short, long = min(w, h), max(w, h)
    if long > 1.7 * short:
        return max(12.0, 0.62 * short)
    return max(HOLD_DIST_PX, HOLD_DIST_SCALE * short)


def hand_obj_hits(probes: list[tuple[float, float]], pts: np.ndarray, thr: float) -> tuple[float, int]:
    if not probes:
        return 1e9, 0
    contour = shrink_obb(pts, HOLD_SHRINK).reshape((-1, 1, 2)).astype(np.float32)
    best = 1e9
    hits = 0
    for x, y in probes:
        d = float(cv2.pointPolygonTest(contour, (float(x), float(y)), True))
        dist = 0.0 if d >= 0 else -d
        best = min(best, dist)
        if dist <= thr:
            hits += 1
    return best, hits


def min_hand_obj_dist(probes: list[tuple[float, float]], pts: np.ndarray) -> float:
    best, _hits = hand_obj_hits(probes, pts, 0.0)
    return best


def box_diag(xyxy: np.ndarray) -> float:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return float(np.hypot(x2 - x1, y2 - y1))


@dataclass
class TrackedItem:
    tid: int
    cls_name: str
    role: str
    name: str
    det: Detection
    pts: np.ndarray
    xyxy: np.ndarray
    center: np.ndarray
    tilt: float


@dataclass
class ActionView:
    held_tids: set[int]
    hole_xyxy: np.ndarray | None
    hud_lines: list[str]
    fired: list[str]
    trace: list[dict] = field(default_factory=list)


class ActionEngine:
    """手持 + 倾倒/倒出 + 四动作候选。不打分、不做拧盖。."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.hold_on: dict[int, int] = defaultdict(int)
        self.hold_off: dict[int, int] = defaultdict(int)
        self.holding: set[int] = set()
        self.last_held_frame: dict[int, int] = {}
        self.tilt_hi: dict[int, int] = defaultdict(int)
        self.tilt_lo: dict[int, int] = defaultdict(int)
        self.pouring: dict[int, bool] = defaultdict(bool)
        self.dumping: dict[int, bool] = defaultdict(bool)
        self.in_hole_n: dict[int, int] = defaultdict(int)
        self.seen_out_hole: dict[int, bool] = defaultdict(bool)
        self.angles: dict[int, deque] = defaultdict(lambda: deque(maxlen=HIST_LEN))
        self.centers: dict[int, deque] = defaultdict(lambda: deque(maxlen=HIST_LEN))
        self.shake_n: dict[int, int] = defaultdict(int)
        self.cut_n = 0
        self.insert_n = 0
        self.first_sterilizer_frame: int | None = None
        self.tweezers_ok = False
        self.knife_ok = False
        self.step1_ok = False
        self.naocl_poured = False
        self.last_pour: str = ""
        self.pour_done_n: dict[str, int] = defaultdict(int)
        self.dump_n = 0
        self.shake_scored = False
        self.shake_unscored = False
        self.cutting_ok = False
        self.insert_ok = False
        self.insert_angle_ok = False
        self.live_pour = ""
        self.live_dump = False
        self.live_shake = ""
        self.missing_note: list[str] = []
        self.trace: list[dict] = []
        self.n_hands = 0
        self.cut_span = 0.0
        self.sterile_shift = 0.0
        self.soaks = {name: self._blank_soak(name) for name in SOAK_SEC}

    def _blank_soak(self, name: str) -> dict:
        target = float(SOAK_SEC[name])
        lo = target * (1.0 - SOAK_MARGIN)
        hi = target * (1.0 + SOAK_MARGIN)
        return {
            "name": name,
            "target": target,
            "lo": lo,
            "hi": hi,
            "started_at": None,
            "last_t": None,
            "elapsed": 0.0,
            "running": False,
            "done": False,
            "early": False,
            "late": False,
            "verdict": "",
            "dump_hold": 0,
            "dump_need": SOAK_DUMP_FRAMES,
            "overtime_warned": False,
            "seen_sterile": False,
        }

    def soak_state(self) -> dict:
        out = {}
        for name, soak in self.soaks.items():
            row = {k: v for k, v in soak.items() if k not in ("origin",)}
            row["elapsed"] = float(soak["elapsed"])
            row["target"] = float(soak["target"])
            row["lo"] = float(soak["lo"])
            row["hi"] = float(soak["hi"])
            row["dump_hold"] = int(soak.get("dump_hold", 0))
            row["dump_need"] = int(SOAK_DUMP_FRAMES)
            row["remain"] = max(0.0, row["target"] - row["elapsed"])
            row["progress"] = min(1.0, row["elapsed"] / row["target"]) if row["target"] else 0.0
            row["window"] = f"{row['lo']:.0f}–{row['hi']:.0f}s"
            out[name] = row
        return out

    def note_pause(self) -> None:
        for soak in self.soaks.values():
            soak["last_t"] = None

    def _sterile_items(self, items: list[TrackedItem]) -> list[TrackedItem]:
        return [it for it in items if it.role == ROLE_STERILE_BOTTLE]

    def _start_soak(self, name: str, items: list[TrackedItem], fired: list[str]) -> None:
        if name not in self.soaks:
            return
        for other, soak in self.soaks.items():
            if other != name and soak["running"]:
                self._stop_soak(other, fired, reason="下一试剂已倒完，上一计时收口")
        soak = self._blank_soak(name)
        now = time.time()
        soak["started_at"] = now
        soak["last_t"] = now
        soak["running"] = True
        soak["seen_sterile"] = bool(self._sterile_items(items))
        self.soaks[name] = soak
        self._fire(
            fired,
            f"计时 {name} 开始 目标{soak['target']:.0f}s 合格窗{soak['lo']:.0f}–{soak['hi']:.0f}s（±10%）"
            + (" 已锁定灭菌瓶" if soak["seen_sterile"] else " 等待锁定灭菌瓶"),
        )

    def _stop_soak(self, name: str, fired: list[str], reason: str = "灭菌瓶倒出液体") -> None:
        soak = self.soaks.get(name)
        if soak is None or not soak["running"]:
            return
        soak["running"] = False
        soak["done"] = True
        soak["last_t"] = None
        t = float(soak["elapsed"])
        if t < soak["lo"]:
            soak["early"] = True
            soak["late"] = False
            soak["verdict"] = "early"
            judge = f"偏早 {t:.1f}s < {soak['lo']:.0f}s"
        elif t > soak["hi"]:
            soak["early"] = False
            soak["late"] = True
            soak["verdict"] = "late"
            judge = f"超时 {t:.1f}s > {soak['hi']:.0f}s"
        else:
            soak["early"] = False
            soak["late"] = False
            soak["verdict"] = "ok"
            judge = f"合格 {t:.1f}s（窗{soak['lo']:.0f}–{soak['hi']:.0f}s）"
        self._fire(fired, f"计时 {name} {reason}停表 {judge}")

    def _tick_soaks(self, items: list[TrackedItem], fired: list[str]) -> None:
        now = time.time()
        steriles = self._sterile_items(items)
        for soak in self.soaks.values():
            if not soak["running"]:
                continue
            if soak["last_t"] is None:
                soak["last_t"] = now
            soak["elapsed"] += max(0.0, now - soak["last_t"])
            soak["last_t"] = now
            if steriles:
                soak["seen_sterile"] = True
            candidates = steriles or [
                it
                for it in items
                if it.cls_name == OCR_CLASS and it.role not in ROLE_REAGENTS and it.tid in self.holding
            ]
            dump_hold = 0
            dump_ready = False
            for it in candidates:
                hold_n, ready = self._dump_progress(it)
                dump_hold = max(dump_hold, hold_n)
                if ready:
                    dump_ready = True
            soak["dump_hold"] = dump_hold
            if soak["elapsed"] >= soak["hi"] and not soak["overtime_warned"]:
                soak["overtime_warned"] = True
                soak["late"] = True
                self._fire(
                    fired,
                    f"计时 {soak['name']} 已超过上限{soak['hi']:.0f}s，仍等待灭菌瓶倒出液体后停表",
                )
            if dump_ready:
                self._stop_soak(soak["name"], fired, reason="灭菌瓶倒出液体")
        self.sterile_shift = 0.0

    def _trace_group(self, gid: str, title: str, summary: str, state: str, steps: list[dict]) -> dict:
        return {"id": gid, "title": title, "summary": summary, "state": state, "steps": steps}

    def _build_trace(
        self,
        items: list[TrackedItem],
        hands: list[dict],
        hole: np.ndarray | None,
        in_opening: bool,
    ) -> list[dict]:
        held = [it for it in items if it.tid in self.holding]
        reagents = [it for it in items if it.role in ROLE_REAGENTS]
        steriles = self._sterile_items(items)
        tweezers = [it for it in items if it.cls_name == CLS_TWEEZER]
        knives = [it for it in items if it.cls_name == CLS_KNIFE]
        sterilizers = [it for it in items if it.cls_name == CLS_STERILIZER]

        def st(ok: bool, warn: bool = False) -> str:
            if ok:
                return "ok"
            return "warn" if warn else "idle"

        hold_steps = [
            {"label": "手部关键点", "value": f"{self.n_hands} 只手", "ok": self.n_hands > 0},
            {
                "label": "防抖",
                "value": f"贴紧{HOLD_ON_FRAMES}帧才算握住，离开{HOLD_OFF_FRAMES}帧才松开",
                "ok": bool(held),
            },
            {"label": "当前手持", "value": "、".join(it.name for it in held) or "无", "ok": bool(held)},
            {
                "label": "距离规则",
                "value": f"缩框后 ≥{HOLD_MIN_HITS} 个手点落入 max({HOLD_DIST_PX:.0f}px, 短边×{HOLD_DIST_SCALE:.2f})；长条工具按短边",
                "ok": bool(held),
            },
        ]
        tw_n = max((self.in_hole_n[it.tid] for it in tweezers), default=0)
        kn_n = max((self.in_hole_n[it.tid] for it in knives), default=0)
        sterile_steps = [
            {"label": "灭菌器", "value": "已检出" if sterilizers else "未检出", "ok": bool(sterilizers)},
            {"label": "孔区", "value": "顶部32%且左右内缩18%" if hole is not None else "无", "ok": hole is not None},
            {"label": "开场已插入", "value": "前90帧在孔内也算，不必先拔出", "ok": in_opening},
            {
                "label": "镊子入孔",
                "value": f"{tw_n}/{IN_HOLE_FRAMES} 帧  "
                + ("完成" if self.tweezers_ok else "进行中" if tw_n else "未入"),
                "ok": self.tweezers_ok,
            },
            {
                "label": "美工刀入孔",
                "value": f"{kn_n}/{IN_HOLE_FRAMES} 帧  " + ("完成" if self.knife_ok else "进行中" if kn_n else "未入"),
                "ok": self.knife_ok,
            },
            {"label": "步骤1", "value": "镊子+美工刀都完成才算灭菌完成", "ok": self.step1_ok},
        ]
        pour_it = next((it for it in reagents if self.pouring[it.tid]), None)
        if pour_it is None and reagents:
            pour_it = max(reagents, key=lambda it: it.tilt, default=None)
        pour_steps = [
            {
                "label": "试剂瓶OCR",
                "value": "、".join(f"{it.role} {it.tilt:.0f}°" for it in reagents) or "未锁定酒精/无菌水/次氯酸钠",
                "ok": bool(reagents),
            },
            {
                "label": "倾倒阈值",
                "value": f"手持且倾角≥{POUR_TILT_DEG:.0f}°，连续{POUR_ON_FRAMES}帧",
                "ok": bool(self.live_pour),
            },
            {
                "label": "回正阈值",
                "value": f"倾角≤{POUR_UPRIGHT_DEG:.0f}°，连续{POUR_OFF_FRAMES}帧 → 计时零点",
                "ok": bool(self.last_pour),
            },
            {
                "label": "当前",
                "value": self.live_pour
                or (f"最近 {self.last_pour} ×{self.pour_done_n[self.last_pour]}" if self.last_pour else "未倾倒"),
                "ok": bool(self.live_pour or self.last_pour),
            },
            {
                "label": "对准灭菌瓶",
                "value": "近距或灭菌瓶在试剂瓶下方才标注→灭菌瓶",
                "ok": any(self._near_sterile(it, items) for it in reagents) if reagents else False,
            },
        ]
        soak_steps = []
        soak_state = "idle"
        soak_summary = "等待倒完回正"
        for name, soak in self.soaks.items():
            if soak["running"]:
                soak_state = "on"
                soak_summary = f"{name} {soak['elapsed']:.1f}/{soak['target']:.0f}s"
            elif soak["done"] and soak_state != "on":
                soak_state = "ok" if soak["verdict"] == "ok" else "warn"
                if soak_summary == "等待倒完回正":
                    soak_summary = f"{name} {soak['verdict'] or '完成'} {soak['elapsed']:.1f}s"
            label = "计时中" if soak["running"] else ("已完成" if soak["done"] else "未开始")
            extra = ""
            if soak["running"] or soak["done"]:
                extra = f" 倒出{int(soak.get('dump_hold', 0))}/{SOAK_DUMP_FRAMES}帧"
                if soak["verdict"] == "early":
                    extra += " 偏早"
                elif soak["verdict"] == "late":
                    extra += " 超时"
                elif soak["verdict"] == "ok":
                    extra += " 合格"
            soak_steps.append(
                {
                    "label": name,
                    "value": f"{label} {soak['elapsed']:.1f}/{soak['target']:.0f}s 窗{soak['lo']:.0f}–{soak['hi']:.0f}s{extra}",
                    "ok": soak["verdict"] == "ok",
                    "warn": soak["verdict"] in ("early", "late") or soak["overtime_warned"],
                }
            )
        soak_steps.append(
            {
                "label": "停表规则",
                "value": f"浸泡中可摇晃；停表看灭菌瓶持续倾倒≥{SOAK_DUMP_TILT:.0f}° 共{SOAK_DUMP_FRAMES}帧，且倾角不再来回换向",
                "ok": any(s["done"] for s in self.soaks.values()),
            }
        )
        dump_steps = [
            {"label": "灭菌瓶OCR", "value": f"{len(steriles)} 个", "ok": bool(steriles)},
            {
                "label": "倒出=停表",
                "value": f"手持灭菌瓶，倾角连续≥{SOAK_DUMP_TILT:.0f}° 满{SOAK_DUMP_FRAMES}帧，换向≤{SOAK_DUMP_MAX_REVERSALS}次（摇晃会换向，倒出是稳住大倾角）",
                "ok": self.live_dump or self.dump_n > 0,
            },
            {"label": "当前", "value": "倒出中" if self.live_dump else "待命", "ok": self.live_dump},
        ]
        next((it.tid for it in steriles if it.tid in self.holding), None)
        shake_steps = [
            {"label": "前置", "value": "必须手持灭菌瓶；次氯酸钠倒完后才计分", "ok": self.naocl_poured},
            {
                "label": "角度",
                "value": f"近{SHAKE_WINDOW}帧极差≥{SHAKE_MIN_RANGE:.0f}° 且标准差≥{SHAKE_ANGLE_STD:.1f}°，至少2次换向",
                "ok": self.shake_scored or self.shake_unscored,
            },
            {"label": "位移上限", "value": f"中心跨度<{SHAKE_MAX_SHIFT:.0f}px（太大算搬走不是摇）", "ok": True},
            {"label": "连续", "value": f"{SHAKE_ON_FRAMES}帧满足才锁定", "ok": self.shake_scored},
            {
                "label": "当前",
                "value": self.live_shake
                or ("计分已检出" if self.shake_scored else ("不计分已检出" if self.shake_unscored else "未检出")),
                "ok": self.shake_scored,
                "warn": self.shake_unscored and not self.shake_scored,
            },
        ]
        cut_steps = [
            {
                "label": "手持美工刀",
                "value": "是" if knives and knives[0].tid in self.holding else "否",
                "ok": bool(knives and knives[0].tid in self.holding),
            },
            {
                "label": "刀心跨度",
                "value": f"{self.cut_span:.0f}/{CUT_SPAN_PX:.0f}px（近{CUT_WINDOW}帧）",
                "ok": self.cut_span >= CUT_SPAN_PX,
            },
            {"label": "连续帧", "value": f"{self.cut_n}/{CUT_HOLD_FRAMES}", "ok": self.cutting_ok},
        ]
        stems = [it for it in items if it.cls_name == "鳞茎"] or [it for it in items if it.cls_name in BULB_CLASSES]
        media = [it for it in items if it.cls_name in MEDIUM_CLASSES or it.role == ROLE_MEDIUM]
        insert_steps = [
            {"label": "鳞茎/种球", "value": "已检出" if stems else "等待检测类", "ok": bool(stems)},
            {"label": "培养基", "value": "已检出" if media else "等待检测类或OCR", "ok": bool(media)},
            {"label": "重叠", "value": f"IoU≥{INSERT_IOU}", "ok": self.insert_ok},
            {
                "label": "夹角",
                "value": f"合格{INSERT_ANGLE_OK[0]:.0f}–{INSERT_ANGLE_OK[1]:.0f}°",
                "ok": self.insert_angle_ok,
            },
        ]
        return [
            self._trace_group(
                "hold", "手持", "、".join(it.name for it in held) or "未持物", "on" if held else "idle", hold_steps
            ),
            self._trace_group(
                "sterile",
                "灭菌",
                "步骤1完成" if self.step1_ok else "镊子/美工刀入孔",
                "ok" if self.step1_ok else ("on" if self.tweezers_ok or self.knife_ok else "idle"),
                sterile_steps,
            ),
            self._trace_group(
                "pour",
                "倾倒",
                self.live_pour or (self.last_pour or "待命"),
                "on" if self.live_pour else ("ok" if self.last_pour else "idle"),
                pour_steps,
            ),
            self._trace_group("soak", "浸泡计时", soak_summary, soak_state if soak_state != "ok" else "ok", soak_steps),
            self._trace_group(
                "dump",
                "倒出",
                f"累计{self.dump_n}次",
                "on" if self.live_dump else ("ok" if self.dump_n else "idle"),
                dump_steps,
            ),
            self._trace_group(
                "shake",
                "摇晃",
                self.live_shake or ("计分" if self.shake_scored else "未检出"),
                "ok" if self.shake_scored else ("warn" if self.shake_unscored else "idle"),
                shake_steps,
            ),
            self._trace_group(
                "cut",
                "切割",
                "已检出" if self.cutting_ok else f"跨度{self.cut_span:.0f}px",
                "ok" if self.cutting_ok else "idle",
                cut_steps,
            ),
            self._trace_group(
                "insert",
                "斜插",
                "已发生" if self.insert_ok else "等待鳞茎/培养基",
                "ok" if self.insert_angle_ok else ("on" if self.insert_ok else "idle"),
                insert_steps,
            ),
        ]

    def _fire(self, fired: list[str], msg: str) -> None:
        fired.append(msg)
        print(f"[动作] {msg}")

    def _recently_held(self, tid: int, frame_id: int) -> bool:
        last = self.last_held_frame.get(tid)
        return last is not None and (frame_id - last) <= RECENT_HOLD_FRAMES

    def _update_holding(self, items: list[TrackedItem], hands: list[dict]) -> None:
        probes_all = [hand_probe_points(h) for h in hands]
        hit: set[int] = set()
        for probes in probes_all:
            if not probes:
                continue
            best_tid, best_d = None, 1e9
            for it in items:
                if it.cls_name in SKIP_HOLD_CLASSES or it.cls_name not in HOLDABLE_CLASSES:
                    continue
                thr = hold_threshold(it)
                dist, n_hit = hand_obj_hits(probes, it.pts, thr)
                gripped = n_hit >= HOLD_MIN_HITS or (n_hit >= 1 and dist <= HOLD_INSIDE_ONE)
                if gripped and dist < best_d:
                    best_tid, best_d = it.tid, dist
            if best_tid is not None:
                hit.add(best_tid)
        seen = {it.tid for it in items}
        for tid in list(self.holding | hit | seen):
            if tid in hit:
                self.hold_on[tid] += 1
                self.hold_off[tid] = 0
                if self.hold_on[tid] >= HOLD_ON_FRAMES:
                    self.holding.add(tid)
            else:
                self.hold_off[tid] += 1
                self.hold_on[tid] = 0
                if self.hold_off[tid] >= HOLD_OFF_FRAMES:
                    self.holding.discard(tid)

    def _near_sterile(self, src: TrackedItem, items: list[TrackedItem]) -> bool:
        for it in items:
            if it.role != ROLE_STERILE_BOTTLE:
                continue
            dist = float(np.linalg.norm(src.center - it.center))
            if dist <= POUR_NEAR_SCALE * 0.5 * (box_diag(src.xyxy) + box_diag(it.xyxy)):
                return True
            if abs(float(it.center[0] - src.center[0])) < 140 and float(it.center[1]) > float(src.center[1]) - 30:
                return True
        return False

    def _is_shaking(self, tid: int) -> bool:
        ang = np.array(self.angles[tid], dtype=np.float32)
        if ang.size < SHAKE_WINDOW:
            return False
        ang = ang[-SHAKE_WINDOW:]
        if float(ang.max() - ang.min()) < SHAKE_MIN_RANGE or float(ang.std()) < SHAKE_ANGLE_STD:
            return False
        pts = np.array(self.centers[tid], dtype=np.float32)[-SHAKE_WINDOW:]
        shift = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        if shift > SHAKE_MAX_SHIFT:
            return False
        d = np.diff(ang)
        if d.size < 2:
            return False
        return int(np.sum((d[1:] * d[:-1]) < 0)) >= 2

    def _dump_progress(self, it: TrackedItem) -> tuple[int, bool]:
        """摇晃是倾角来回换向；倒出是灭菌瓶持续保持大倾角。返回 (已连续高倾角帧数, 是否判定为倒出停表)。."""
        if it.tid not in self.holding:
            return 0, False
        ang = np.array(self.angles.get(it.tid, []), dtype=np.float32)
        if ang.size == 0:
            return 0, False
        hold_n = 0
        for a in ang[::-1]:
            if float(a) >= SOAK_DUMP_TILT:
                hold_n += 1
            else:
                break
        n = SOAK_DUMP_FRAMES
        if hold_n < n or ang.size < n:
            return hold_n, False
        recent = ang[-n:]
        if float(recent.min()) < SOAK_DUMP_TILT:
            return hold_n, False
        d = np.diff(recent)
        reversals = 0
        if d.size >= 2:
            reversals = int(np.sum((d[1:] * d[:-1]) < 0))
        if reversals > SOAK_DUMP_MAX_REVERSALS:
            return hold_n, False
        return hold_n, True

    def update(
        self,
        frame_id: int,
        items: list[TrackedItem],
        hands: list[dict],
        class_names: dict[int, str],
    ) -> ActionView:
        fired: list[str] = []
        self.live_pour = ""
        self.live_dump = False
        self.live_shake = ""
        self.cut_span = 0.0
        self.n_hands = len(hands)
        names = set(class_names.values())
        self.missing_note = []
        if not any(c in names for c in BULB_CLASSES):
            self.missing_note.append("斜插等待种球·鳞茎类别")
        if not any(c in names for c in MEDIUM_CLASSES) and not any(it.role == ROLE_MEDIUM for it in items):
            self.missing_note.append("斜插等待培养基类别或OCR")

        self._update_holding(items, hands)
        for it in items:
            if it.tid in self.holding:
                self.last_held_frame[it.tid] = frame_id
            self.angles[it.tid].append(it.tilt)
            self.centers[it.tid].append(it.center.copy())

        sterilizers = [it for it in items if it.cls_name == CLS_STERILIZER]
        hole = hole_xyxy_from_sterilizer(sterilizers[0].xyxy) if sterilizers else None
        if sterilizers and self.first_sterilizer_frame is None:
            self.first_sterilizer_frame = frame_id
        in_opening = (
            hole is not None
            and self.first_sterilizer_frame is not None
            and (frame_id - self.first_sterilizer_frame) < OPENING_FRAMES
        )

        for it in items:
            if it.cls_name not in (CLS_TWEEZER, CLS_KNIFE) or hole is None:
                continue
            inside = in_hole(it, hole)
            if inside:
                self.in_hole_n[it.tid] += 1
            else:
                self.in_hole_n[it.tid] = 0
                self.seen_out_hole[it.tid] = True
            ready = self.in_hole_n[it.tid] >= IN_HOLE_FRAMES
            held = it.tid in self.holding or self._recently_held(it.tid, frame_id)
            entered = self.seen_out_hole[it.tid] and inside and held
            if not ready:
                continue
            if in_opening or entered or (inside and held):
                if it.cls_name == CLS_TWEEZER and not self.tweezers_ok:
                    self.tweezers_ok = True
                    self._fire(fired, f"灭菌 镊子 完成 frame={frame_id}")
                if it.cls_name == CLS_KNIFE and not self.knife_ok:
                    self.knife_ok = True
                    self._fire(fired, f"灭菌 美工刀 完成 frame={frame_id}")
        if self.tweezers_ok and self.knife_ok and not self.step1_ok:
            self.step1_ok = True
            self._fire(fired, f"步骤1灭菌完成 frame={frame_id}")

        for it in items:
            if it.cls_name != OCR_CLASS:
                continue
            held = it.tid in self.holding
            if it.role in ROLE_REAGENTS:
                if held and it.tilt >= POUR_TILT_DEG:
                    self.tilt_hi[it.tid] += 1
                    self.tilt_lo[it.tid] = 0
                    if self.tilt_hi[it.tid] >= POUR_ON_FRAMES:
                        if not self.pouring[it.tid]:
                            self.pouring[it.tid] = True
                            extra = ""
                            if self._near_sterile(it, items):
                                extra = "→灭菌瓶"
                            elif not any(x.role == ROLE_STERILE_BOTTLE for x in items):
                                extra = "（未锁定灭菌瓶）"
                            self._fire(fired, f"开始倾倒 {it.role}{extra} frame={frame_id}")
                        self.live_pour = f"倾倒中 {it.role}"
                elif self.pouring[it.tid] and it.tilt <= POUR_UPRIGHT_DEG:
                    self.tilt_lo[it.tid] += 1
                    self.tilt_hi[it.tid] = 0
                    if self.tilt_lo[it.tid] >= POUR_OFF_FRAMES:
                        self.pouring[it.tid] = False
                        self.pour_done_n[it.role] += 1
                        self.last_pour = it.role
                        if it.role == "次氯酸钠":
                            self.naocl_poured = True
                        self._fire(fired, f"倒完回正 {it.role} 计时零点 frame={frame_id}")
                        self._start_soak(it.role, items, fired)
                elif self.pouring[it.tid]:
                    self.tilt_hi[it.tid] = 0
                    self.live_pour = f"倾倒中 {it.role}"
                elif not held:
                    self.tilt_hi[it.tid] = 0
            elif it.role == ROLE_STERILE_BOTTLE:
                if held and it.tilt >= POUR_TILT_DEG:
                    self.tilt_hi[it.tid] += 1
                    self.tilt_lo[it.tid] = 0
                    if self.tilt_hi[it.tid] >= POUR_ON_FRAMES:
                        if not self.dumping[it.tid]:
                            self.dumping[it.tid] = True
                            self.dump_n += 1
                            self._fire(fired, f"倒出 灭菌瓶 第{self.dump_n}次 frame={frame_id}")
                        self.live_dump = True
                elif self.dumping[it.tid] and it.tilt <= POUR_UPRIGHT_DEG:
                    self.tilt_lo[it.tid] += 1
                    self.tilt_hi[it.tid] = 0
                    if self.tilt_lo[it.tid] >= POUR_OFF_FRAMES:
                        self.dumping[it.tid] = False
                elif self.dumping[it.tid]:
                    self.live_dump = True
                elif not held:
                    self.tilt_hi[it.tid] = 0

        for it in items:
            if it.role != ROLE_STERILE_BOTTLE or it.tid not in self.holding:
                self.shake_n[it.tid] = 0
                continue
            if self._is_shaking(it.tid):
                self.shake_n[it.tid] += 1
            else:
                self.shake_n[it.tid] = 0
            if self.shake_n[it.tid] < SHAKE_ON_FRAMES:
                continue
            if self.naocl_poured:
                self.live_shake = "摇晃(计分)"
                if not self.shake_scored:
                    self.shake_scored = True
                    self._fire(fired, f"摇晃 计分 frame={frame_id}")
            else:
                self.live_shake = "摇晃(不计分)"
                if not self.shake_unscored:
                    self.shake_unscored = True
                    self._fire(fired, f"摇晃 不计分（未完成次氯酸钠倒入） frame={frame_id}")

        knives = [it for it in items if it.cls_name == CLS_KNIFE]
        if knives:
            knife = knives[0]
            moving = False
            hist = np.array(self.centers.get(knife.tid, []), dtype=np.float32)
            if hist.shape[0] >= CUT_WINDOW:
                hist = hist[-CUT_WINDOW:]
                span = float(np.linalg.norm(hist.max(axis=0) - hist.min(axis=0)))
                self.cut_span = span
                moving = span >= CUT_SPAN_PX
            if knife.tid in self.holding and moving:
                self.cut_n += 1
            else:
                self.cut_n = 0
            if self.cut_n >= CUT_HOLD_FRAMES and not self.cutting_ok:
                self.cutting_ok = True
                self._fire(fired, f"切割 完成 frame={frame_id}")
        else:
            self.cut_n = 0

        stems = [it for it in items if it.cls_name == "鳞茎"] or [it for it in items if it.cls_name in BULB_CLASSES]
        media = [it for it in items if it.cls_name in MEDIUM_CLASSES or it.role == ROLE_MEDIUM]
        if stems and media:
            stem = stems[0]
            med = min(media, key=lambda m: float(np.linalg.norm(stem.center - m.center)))
            overlapped = xyxy_iou(stem.xyxy, med.xyxy) >= INSERT_IOU
            held = stem.tid in self.holding or self._recently_held(stem.tid, frame_id)
            if overlapped and held:
                self.insert_n += 1
            else:
                self.insert_n = 0
            if self.insert_n >= 6:
                ang = axes_included_deg(stem.pts, med.pts)
                if not self.insert_ok:
                    self.insert_ok = True
                    self._fire(fired, f"斜插 发生 夹角{ang:.0f}° frame={frame_id}")
                if INSERT_ANGLE_OK[0] <= ang <= INSERT_ANGLE_OK[1] and not self.insert_angle_ok:
                    self.insert_angle_ok = True
                    self._fire(fired, f"斜插 角度合格 {ang:.0f}° frame={frame_id}")
        else:
            self.insert_n = 0

        self._tick_soaks(items, fired)

        held_names = [it.name for it in items if it.tid in self.holding]
        tw = "✓" if self.tweezers_ok else "○"
        kn = "✓" if self.knife_ok else "○"
        hud = [
            "手持: " + ("、".join(held_names) if held_names else "无") + f"  手点数{self.n_hands}",
            f"灭菌: 镊子{tw} 美工刀{kn}" + ("  步骤1完成" if self.step1_ok else ""),
        ]
        if self.live_pour:
            hud.append(self.live_pour)
        elif self.last_pour:
            hud.append(f"最近倒完: {self.last_pour} ×{self.pour_done_n[self.last_pour]}")
        running = next((s for s in self.soaks.values() if s["running"]), None)
        if running:
            hud.append(
                f"浸泡{running['name']} {running['elapsed']:.1f}/{running['target']:.0f}s "
                f"窗{running['lo']:.0f}–{running['hi']:.0f} 倒出{int(running.get('dump_hold', 0))}/{SOAK_DUMP_FRAMES}帧"
            )
        if self.live_dump:
            hud.append(f"倒出中 累计{self.dump_n}次")
        elif self.dump_n:
            hud.append(f"倒出累计 {self.dump_n}次")
        if self.live_shake:
            hud.append(self.live_shake)
        elif self.shake_scored:
            hud.append("摇晃: 计分已检出")
        elif self.shake_unscored:
            hud.append("摇晃: 已检出(不计分)")
        hud.append(
            f"切割: {'已检出' if self.cutting_ok else '未检出'} 跨度{self.cut_span:.0f}/{CUT_SPAN_PX:.0f}px 连续{self.cut_n}/{CUT_HOLD_FRAMES}"
        )
        if self.insert_ok:
            hud.append("斜插: 已发生" + (" 角度合格" if self.insert_angle_ok else " 角度待核"))
        elif not any(c in names for c in BULB_CLASSES):
            hud.append("斜插: 等待鳞茎/种球类别")
        elif not (any(c in names for c in MEDIUM_CLASSES) or any(it.role == ROLE_MEDIUM for it in items)):
            hud.append("斜插: 等待培养基OCR或类别")
        else:
            hud.append("斜插: 未检出")
        self.trace = self._build_trace(items, hands, hole, in_opening)
        return ActionView(set(self.holding), hole, hud, fired, self.trace)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实时：手套 RTMPose + 瓶子追踪 OCR + 动作规则")
    p.add_argument("--source", default="0", help="摄像头编号或视频路径")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO-OBB 权重")
    p.add_argument(
        "--hand-mode",
        choices=("glove", "bare"),
        default="glove",
        help="glove=戴手套(YOLO手套框)  bare=不戴手套(RTMDet全图检手)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--pose-conf", type=float, default=0.25)
    p.add_argument("--det-thr", type=float, default=0.25)
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"找不到 YOLO 权重: {weights}")

    model = YOLO(str(weights))
    class_names: dict[int, str] = model.names
    print(f"YOLO 类别: {list(class_names.values())}")
    if HAND_CLASS not in class_names.values():
        print(f"警告: 模型没有「{HAND_CLASS}」。戴手套模式下骨架可能出不来，可按 H 切到不戴手套。")
    if OCR_CLASS not in class_names.values():
        print(f"警告: 模型没有「{OCR_CLASS}」，不会触发 OCR。")

    ocr = load_ocr()
    tracker = load_bytetrack(TRACK_BUFFER)
    tracker.args.match_thresh = TRACK_MATCH_THRESH
    ocr_cache: dict[int, OcrRecord] = {}
    last_seen: dict[int, int] = {}
    ocr_bind_mem: dict = {}
    engine = ActionEngine()

    rtm_dev = yolo_device_to_rtm(args.device)
    rtm_det, pose = make_rtm(rtm_dev, args.det_thr)
    cap, is_cam = open_source(args.source)
    hand_mode = args.hand_mode
    paused = False
    frame_id = 0
    snap_dir = ROOT / "runs" / "detect"
    snap_dir.mkdir(parents=True, exist_ok=True)

    print("实时监测：瓶子走追踪 OCR，手部走 RTMPose，动作走几何规则。")
    print("按 Q 退出  S 保存  H 切换戴手套/不戴手套  R 重置动作  空格暂停")
    print(f"当前手模式: {'戴手套(YOLO手套框)' if hand_mode == 'glove' else '不戴手套(RTMDet全图)'}")

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

        orig = frame
        result = model.predict(
            source=orig,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        dets = parse_detections(result)
        active, _removed = update_tracks(tracker, dets, orig)

        active_ids = {tid for tid, _, _ in active}
        do_ocr = frame_id % max(OCR_EVERY, 1) == 0
        gloves = [
            (class_names.get(d.cls_id, str(d.cls_id)), d.conf, d.xyxy, d.pts)
            for d in dets
            if class_names.get(d.cls_id, str(d.cls_id)) == HAND_CLASS
        ]
        tracked_glove_xyxy: list[list[float]] = []
        items: list[TrackedItem] = []
        box_rows: list[tuple[np.ndarray, int, str, float, int]] = []
        bottle_obs: list[tuple[int, np.ndarray, float]] = []
        for tid, det, _track_score in active:
            last_seen[tid] = frame_id
            if class_names.get(det.cls_id, str(det.cls_id)) == OCR_CLASS:
                bottle_obs.append(
                    (
                        tid,
                        det.pts.mean(axis=0),
                        float(max(det.xywh[2], det.xywh[3])),
                        np.asarray(det.xyxy, dtype=np.float32),
                    )
                )
        for src, dst, lab in rebind_ocr_cache(
            ocr_cache,
            bottle_obs,
            frame_id,
            last_seen,
            OCR_INHERIT_DIST,
            OCR_INHERIT_GAP,
            OCR_STICKY_DIST,
            ocr_bind_mem,
            OCR_HOLD_FRAMES,
        ):
            print(f"[OCR] 绑定 id={src} → {dst}  {bottle_label(lab)}")
        close_ids = crowded_bottle_tids(bottle_obs)
        blocked_ids = ocr_blocked_tids(ocr_bind_mem, frame_id)

        for tid, det, track_score in active:
            cls_name = class_names.get(det.cls_id, str(det.cls_id))
            center = det.pts.mean(axis=0)
            name = cls_name
            role = ""
            size = float(max(det.xywh[2], det.xywh[3]))

            if cls_name == HAND_CLASS:
                tracked_glove_xyxy.append([float(v) for v in det.xyxy])

            if cls_name == OCR_CLASS:
                record = ocr_cache.get(tid)
                crowded = tid in close_ids
                if (
                    (not crowded)
                    and (tid not in blocked_ids)
                    and should_run_ocr(record, frame_id, do_ocr, OCR_RETRY, OCR_REFRESH)
                ):
                    expand = neighbor_crop_expand(center, size, bottle_obs, tid, CROP_EXPAND)
                    if det.pts is not None and det.pts.shape == (4, 2):
                        crop = crop_obb(orig, det.pts, expand=expand)
                    else:
                        crop = crop_aabb(orig, det.xyxy, expand=expand)
                    text, score = run_ocr(ocr, crop) if crop is not None else ("", 0.0)
                    matched = match_reagent_label(text)
                    if neighbor_label_conflict(tid, matched, ocr_cache, bottle_obs):
                        matched = ""
                    old_lab = record.reagent if record is not None else ""
                    new_rec = OcrRecord(
                        raw_text=matched,
                        score=score,
                        reagent=matched,
                        center=center.copy(),
                        frame_id=frame_id,
                        ocr_frame=frame_id,
                        size=size,
                        height=float(max(det.xyxy[3] - det.xyxy[1], 1.0)),
                    )
                    if record is not None:
                        new_rec.vel = np.asarray(record.vel, dtype=np.float32).copy()
                    ocr_cache[tid] = merge_ocr_cache(record, new_rec)
                    if matched and matched != old_lab:
                        print(f"[OCR] id={tid}  {text!r} -> {bottle_label(matched)}")
                    elif text and not matched:
                        print(f"[OCR] id={tid}  {text!r} -> 未匹配，沿用{bottle_label(old_lab)}")
                rec_now = ocr_cache.get(tid, OcrRecord())
                role = rec_now.reagent
                name = bottle_label(rec_now.raw_text)

            items.append(
                TrackedItem(
                    tid=tid,
                    cls_name=cls_name,
                    role=role,
                    name=name,
                    det=det,
                    pts=det.pts,
                    xyxy=det.xyxy,
                    center=center.astype(np.float32),
                    tilt=obb_tilt_from_vertical(det.pts),
                )
            )
            box_rows.append((det.pts, det.cls_id, name, track_score, tid))

        expire_before = frame_id - OCR_HOLD_FRAMES
        for cache_tid in list(ocr_cache):
            if cache_tid in active_ids:
                continue
            if last_seen.get(cache_tid, 0) <= expire_before:
                ocr_cache.pop(cache_tid, None)
                last_seen.pop(cache_tid, None)

        pose_boxes = glove_boxes_from_yolo(gloves, orig.shape) if hand_mode == "glove" else boxes_to_list(rtm_det(orig))
        hands = pose_on_boxes(pose, orig, pose_boxes, args.pose_conf)
        view = engine.update(frame_id, items, hands, class_names)

        vis = np.ascontiguousarray(orig.copy())
        annotator = Annotator(vis, example="无菌水瓶灭菌摇晃切割斜插戴手套")
        for pts, cls_id, name, track_score, tid in box_rows:
            tag = f"持 {name} {track_score:.2f}" if tid in view.held_tids else f"{name} {track_score:.2f}"
            annotator.box_label(pts, tag, color=colors(cls_id, True))
        glove_cls = next((i for i, n in class_names.items() if n == HAND_CLASS), 0)
        if hand_mode == "glove":
            for _name, conf, xyxy, pts in gloves:
                xy = [float(v) for v in xyxy]
                if any(iou_xyxy(xy, t) >= 0.3 for t in tracked_glove_xyxy):
                    continue
                annotator.box_label(pts, f"{HAND_CLASS} {conf:.2f}", color=colors(glove_cls, True))
        hand_mode_txt = "手模式: 戴手套" if hand_mode == "glove" else "手模式: 不戴手套"
        annotator.text([10, 18], hand_mode_txt, txt_color=(0, 255, 180))
        for i, line in enumerate(view.hud_lines):
            annotator.text([10, 40 + i * 22], line, txt_color=(0, 255, 255))
        vis = np.ascontiguousarray(np.array(annotator.result(), copy=True))
        if view.hole_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in view.hole_xyxy]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 1, cv2.LINE_AA)
        for hand in hands:
            draw_hand(vis, hand["pts"])

        cv2.imshow("Hand + Track + OCR + Action", vis)
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            snap = snap_dir / f"hand_ocr_{frame_id:06d}.jpg"
            cv2.imwrite(str(snap), vis)
            print(f"已保存 {snap}")
        if key in (ord("h"), ord("H"), ord("m"), ord("M")):
            hand_mode = "bare" if hand_mode == "glove" else "glove"
            print(f"切换到手模式: {'戴手套(YOLO手套框)' if hand_mode == 'glove' else '不戴手套(RTMDet全图)'}")
        if key in (ord("r"), ord("R")):
            engine.reset()
            print("已重置动作状态")
        if key == 32:
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
