"""实时监测：YOLO-OBB + ByteTrack + 瓶子 OCR + 手套 RTMPose 21 点。

本文件独立运行，不依赖项目里其它 .py。
画面上不叠检出率、漏检、抖动等统计字。

用法:
    python video_track_ocr_hand.py
    python video_track_ocr_hand.py --source 0
    python video_track_ocr_hand.py --source "视频.mp4"

按键: Q 退出  S 保存  M 切换手套框来源  空格暂停
"""
from __future__ import annotations

import argparse
import os
import sys
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
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
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

# MediaPipe 同款 21 点连线，避免再依赖 hand_mediapipe.py
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)
PALM_COLOR = (240, 240, 240)
FINGER_COLORS = {
    1: (40, 140, 255), 2: (40, 140, 255), 3: (40, 140, 255), 4: (40, 140, 255),
    5: (0, 220, 80), 6: (0, 220, 80), 7: (0, 220, 80), 8: (0, 220, 80),
    9: (0, 220, 255), 10: (0, 220, 255), 11: (0, 220, 255), 12: (0, 220, 255),
    13: (220, 80, 255), 14: (220, 80, 255), 15: (220, 80, 255), 16: (220, 80, 255),
    17: (0, 255, 255), 18: (0, 255, 255), 19: (0, 255, 255), 20: (0, 255, 255),
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
    providers = list(getattr(pose, "session").get_providers())
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="实时：手套 RTMPose + 瓶子追踪 OCR")
    p.add_argument("--source", default="0", help="摄像头编号或视频路径")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO-OBB 权重")
    p.add_argument("--mode", choices=("roi", "full"), default="roi")
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
        print(f"警告: 模型没有「{HAND_CLASS}」，手套骨架可能出不来。")
    if OCR_CLASS not in class_names.values():
        print(f"警告: 模型没有「{OCR_CLASS}」，不会触发 OCR。")

    ocr = load_ocr()
    tracker = load_bytetrack(TRACK_BUFFER)
    tracker.args.match_thresh = TRACK_MATCH_THRESH
    ocr_cache: dict[int, OcrRecord] = {}
    last_seen: dict[int, int] = {}
    ocr_bind_mem: dict = {}

    rtm_dev = yolo_device_to_rtm(args.device)
    rtm_det, pose = make_rtm(rtm_dev, args.det_thr)
    cap, is_cam = open_source(args.source)
    mode = args.mode
    paused = False
    frame_id = 0
    snap_dir = ROOT / "runs" / "detect"
    snap_dir.mkdir(parents=True, exist_ok=True)

    print("实时监测：瓶子走追踪 OCR，手套走 RTMPose。")
    print("按 Q 退出  S 保存  M 切换手套框来源  空格暂停")

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

        vis = np.ascontiguousarray(orig.copy())
        annotator = Annotator(vis, example="无菌水瓶")
        active_ids = {tid for tid, _, _ in active}
        do_ocr = frame_id % max(OCR_EVERY, 1) == 0
        gloves = [
            (class_names.get(d.cls_id, str(d.cls_id)), d.conf, d.xyxy, d.pts)
            for d in dets
            if class_names.get(d.cls_id, str(d.cls_id)) == HAND_CLASS
        ]
        tracked_glove_xyxy: list[list[float]] = []
        bottle_obs: list[tuple[int, np.ndarray, float]] = []
        for tid, det, _track_score in active:
            last_seen[tid] = frame_id
            if class_names.get(det.cls_id, str(det.cls_id)) == OCR_CLASS:
                bottle_obs.append(
                    (tid, det.pts.mean(axis=0), float(max(det.xywh[2], det.xywh[3])), np.asarray(det.xyxy, dtype=np.float32))
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
            size = float(max(det.xywh[2], det.xywh[3]))

            if cls_name == HAND_CLASS:
                tracked_glove_xyxy.append([float(v) for v in det.xyxy])

            if cls_name == OCR_CLASS:
                record = ocr_cache.get(tid)
                if (tid not in close_ids) and (tid not in blocked_ids) and should_run_ocr(record, frame_id, do_ocr, OCR_RETRY, OCR_REFRESH):
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
                name = bottle_label(ocr_cache.get(tid, OcrRecord()).raw_text)

            annotator.box_label(det.pts, f"{name} {track_score:.2f}", color=colors(det.cls_id, True))

        glove_cls = next((i for i, n in class_names.items() if n == HAND_CLASS), 0)
        for _name, conf, xyxy, pts in gloves:
            xy = [float(v) for v in xyxy]
            if any(iou_xyxy(xy, t) >= 0.3 for t in tracked_glove_xyxy):
                continue
            annotator.box_label(pts, f"{HAND_CLASS} {conf:.2f}", color=colors(glove_cls, True))

        expire_before = frame_id - OCR_HOLD_FRAMES
        for cache_tid in list(ocr_cache):
            if cache_tid in active_ids:
                continue
            if last_seen.get(cache_tid, 0) <= expire_before:
                ocr_cache.pop(cache_tid, None)
                last_seen.pop(cache_tid, None)

        vis = np.ascontiguousarray(np.array(annotator.result(), copy=True))

        pose_boxes = glove_boxes_from_yolo(gloves, orig.shape)
        if mode == "full":
            pose_boxes = merge_boxes(pose_boxes, boxes_to_list(rtm_det(orig)))
        for hand in pose_on_boxes(pose, orig, pose_boxes, args.pose_conf):
            draw_hand(vis, hand["pts"])

        cv2.imshow("Hand + Track + OCR", vis)
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            snap = snap_dir / f"hand_ocr_{frame_id:06d}.jpg"
            cv2.imwrite(str(snap), vis)
            print(f"已保存 {snap}")
        if key in (ord("m"), ord("M")):
            mode = "full" if mode == "roi" else "roi"
            print(f"切换到 {'YOLO手套+RTMDet补漏' if mode == 'full' else '仅YOLO手套框'}")
        if key == 32:
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
