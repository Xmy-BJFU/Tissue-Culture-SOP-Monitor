"""实时摄像头：YOLO-OBB + ByteTrack + OCR。

框上只保留一条 YOLO 风格标签「名字 + 置信度」。
OCR 归到酒精 / 无菌水 / 次氯酸钠 / 灭菌瓶 / 培养基。

用法:
    python video_track_ocr.py

按 Q 退出，按 S 保存当前帧。
"""
from __future__ import annotations

import sys
import time
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
    match_reagent_label,
    neighbor_crop_expand,
    ocr_blocked_tids,
    rebind_ocr_cache,
    should_run_ocr,
)

WEIGHTS = r"E:\XMY\代码\ultralytics\runs\train\11n_100_deg45（26_17）\weights\best.pt"
CAMERA_ID = 0
IMGSZ = 640              # YOLO 检测输入边长，只影响找框准不准/快不快，不加 OCR 清晰度
CONF = 0.25              # YOLO 置信度阈值，低于此的检测丢掉
IOU = 0.7                # YOLO NMS 的 IoU 阈值，越大越容易保留重叠框
DEVICE = "0"             # YOLO 设备：0=第一块 GPU，cpu=CPU
OCR_CLASS = "瓶子"        # 只有这个检测类才做 OCR，其它类只画「类别 + 置信度」

# ---- OCR 节奏（单位都是「处理帧」，秒数 ≈ 帧数 / 画面 FPS）----
OCR_EVERY = 4            # 每隔 N 帧才允许跑 OCR，避免每帧都打满 CPU
OCR_RETRY = 6            # 还没归到已知标签时，隔多少帧再试
OCR_REFRESH = 18         # 已经有标签时，隔多少帧复核一次；检到不同则改用新标签，没检到则沿用
MOVE_THRESH = 40.0       # 保留兼容，不再用位移强制 OCR（快移时画面糊，强制识别易把标签冲掉）

# ---- 裁图后再送给 OCR ----
CROP_EXPAND = 1.12       # YOLO 框再放大的倍数，避免字贴边被切掉；太大易带进背景
MIN_CROP_SIDE = 48       # 裁图最短边小于此则放大（插值，远处糊字不会因此变清晰）
MAX_CROP_SIDE = 960      # 裁图最长边上限，防止图太大把 CPU 卡死

# ---- PaddleOCR：先找字再认字 ----
DET_THRESH = 0.3         # 文字检测像素阈值，越低越容易框到淡字，也更容易假框
DET_BOX_THRESH = 0.6     # 文字框置信度，低于此丢掉
DET_UNCLIP = 1.5         # 文字框外扩，笔画连在一起时可略加大
REC_SCORE_THRESH = 0.5   # 单行识别分低于此丢掉；手写/远距离可降到 0.3，误识别也会变多

# ---- 框还在不在 / OCR 结果还在不在（两套独立计时）----
TRACK_BUFFER = 150       # ByteTrack：跟丢后 ID 还活多少帧
TRACK_MATCH_THRESH = 0.85 # ByteTrack 代价=1-IoU，越大越能跟上快移；近距串标由 OCR 绑定层拦截
OCR_HOLD_FRAMES = 180    # 框消失后 OCR 名字再保留多少帧
OCR_INHERIT_DIST = 140.0 # 预测位置匹配的最大距离（快移换 ID）
OCR_INHERIT_GAP = 48.0   # 最近与次近差距不够则不继承，避免两瓶靠太近串标
OCR_STICKY_DIST = 52.0   # 当前 ID 仍对准原瓶时优先粘住；交叉跳变时不粘，按预测位置改绑



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


def load_bytetrack(track_buffer: int) -> BYTETracker:
    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("bytetrack.yaml")))
    cfg.track_buffer = track_buffer
    cfg.match_thresh = TRACK_MATCH_THRESH
    tracker = BYTETracker(args=cfg)
    print(
        f"ByteTrack 就绪  high={cfg.track_high_thresh}  low={cfg.track_low_thresh}  "
        f"buffer={cfg.track_buffer}  match={cfg.match_thresh}"
    )
    return tracker


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
    xywh = np.stack([d.xywh for d in dets], axis=0)
    conf = np.array([d.conf for d in dets], dtype=np.float32)
    cls = np.array([d.cls_id for d in dets], dtype=np.float32)
    return AABBDetections(xywh, conf, cls)


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


def open_camera(camera_id: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"打不开摄像头 {camera_id}。请关掉占用摄像头的软件后重试。")
    return cap


def bottle_label(ocr_text: str) -> str:
    """酒精瓶 / 无菌水瓶 / 次氯酸钠瓶 / 灭菌瓶 / 培养基；对不上仍显示「瓶子」。"""
    name = match_reagent_label(ocr_text)
    if not name:
        return OCR_CLASS
    if name in ("灭菌瓶", "培养基"):
        return name
    return f"{name}瓶"


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


def main() -> None:
    weights = Path(WEIGHTS)
    if not weights.exists():
        raise FileNotFoundError(f"找不到 YOLO 权重: {weights}")

    model = YOLO(str(weights))
    class_names: dict[int, str] = model.names
    print(f"YOLO 类别: {class_names}")
    if OCR_CLASS not in class_names.values():
        print(
            f"警告: 模型类别里没有「{OCR_CLASS}」，不会触发 OCR。"
            f"当前类别: {list(class_names.values())}"
        )

    ocr = load_ocr()
    tracker = load_bytetrack(TRACK_BUFFER)
    tracker.args.match_thresh = TRACK_MATCH_THRESH
    ocr_cache: dict[int, OcrRecord] = {}
    last_seen: dict[int, int] = {}
    ocr_bind_mem: dict = {}
    cap = open_camera(CAMERA_ID)
    print("摄像头已打开，按 Q 退出，按 S 保存当前帧")
    print("框上只显示：名字 + 置信度（瓶子有 OCR 时为「无菌水瓶」这类合成名）")

    frame_id = 0
    t0 = time.time()
    fps_show = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读帧失败")
            break
        frame_id += 1
        orig = frame.copy()

        result = model.predict(
            source=orig,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False,
        )[0]
        dets = parse_detections(result)
        active, _removed_ids = update_tracks(tracker, dets, orig)

        vis = np.ascontiguousarray(orig.copy())
        annotator = Annotator(vis, example="无菌水瓶")
        active_ids = {tid for tid, _, _ in active}
        do_ocr = frame_id % max(OCR_EVERY, 1) == 0
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

            annotator.box_label(
                det.pts,
                f"{name} {track_score:.2f}",
                color=colors(det.cls_id, True),
            )

        expire_before = frame_id - OCR_HOLD_FRAMES
        for cache_tid in list(ocr_cache):
            if cache_tid in active_ids:
                continue
            if last_seen.get(cache_tid, 0) <= expire_before:
                ocr_cache.pop(cache_tid, None)
                last_seen.pop(cache_tid, None)

        vis = np.ascontiguousarray(np.array(annotator.result(), copy=True))
        if frame_id % 10 == 0:
            now = time.time()
            fps_show = 10.0 / max(now - t0, 1e-6)
            t0 = now
        cv2.putText(
            vis,
            f"FPS {fps_show:.1f}",
            (12, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLO-OBB + Track + OCR", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            snap = Path(r"E:\XMY\代码\ultralytics\runs\detect") / f"track_ocr_{frame_id:06d}.jpg"
            snap.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snap), vis)
            print(f"已保存 {snap}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
