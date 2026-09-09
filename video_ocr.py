"""外接/本机摄像头实时 YOLO-OBB 检测 + 轻量 OCR。

和 PP-OCRv6 在线 demo 的 tiny 模型对齐：
    PP-OCRv6_tiny_det + PP-OCRv6_tiny_rec
裁图按画面「上」摆正四点，不做 90° 强转，立着的瓶子上横排字保持从左到右。

用法:
    python video_ocr.py

按 Q 退出，按 S 保存当前帧（同时会把 OCR 裁图存下来，方便和在线 demo 对比）。
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ocr_track_bind import (
    assign_spatial_tids,
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
IMGSZ = 640
CONF = 0.25
IOU = 0.7
DEVICE = "0"
OCR_CLASS = "瓶子"

# ---- OCR 超参：尽量贴近在线 demo 默认，不要乱降阈值 ----
OCR_EVERY = 4          # 每隔 N 帧才允许跑 OCR
OCR_RETRY = 6          # 还没归到已知标签时，隔多少帧再试
OCR_REFRESH = 18       # 已有标签时隔多少帧复核；检到不同则改用新标签，没检到则沿用
OCR_HOLD_FRAMES = 180
OCR_INHERIT_DIST = 140.0
OCR_INHERIT_GAP = 40.0
OCR_STICKY_DIST = 110.0
CROP_EXPAND = 1.12     # 裁框略放大，避免贴边切字
MAX_CROP_SIDE = 960    # 不要压太小；在线 demo 用的是原图清晰度
MIN_CROP_SIDE = 48     # 只在裁图过小时才放大
DET_THRESH = 0.3       # 官方默认约 0.3，过低会出一堆假框
DET_BOX_THRESH = 0.6   # 官方默认约 0.6
DET_UNCLIP = 1.5       # 官方默认约 1.5
REC_SCORE_THRESH = 0.5 # 官方默认约 0.5；仍滤掉就降到 0.3


@dataclass
class OcrSlot:
    raw_text: str = ""
    score: float = 0.0
    reagent: str = ""
    center: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    size: float = 0.0
    height: float = 0.0
    frame_id: int = 0
    ocr_frame: int = 0


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


def _iter_dets(result):
    dets = []
    if result.obb is not None and len(result.obb):
        xyxyxyxy = result.obb.xyxyxyxy.cpu().numpy()
        confs = result.obb.conf.cpu().numpy()
        clss = result.obb.cls.cpu().numpy().astype(int)
        for pts, conf, cls_id in zip(xyxyxyxy, confs, clss):
            dets.append(
                {
                    "kind": "obb",
                    "pts": pts.reshape(4, 2),
                    "xyxy": None,
                    "conf": float(conf),
                    "cls": int(cls_id),
                }
            )
        return dets
    if result.boxes is not None and len(result.boxes):
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(xyxy, confs, clss):
            x1, y1, x2, y2 = box
            pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
            dets.append(
                {
                    "kind": "aabb",
                    "pts": pts,
                    "xyxy": box,
                    "conf": float(conf),
                    "cls": int(cls_id),
                }
            )
    return dets


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

    print("正在加载 PP-OCRv6_tiny（与在线 demo 同一档，首次会下载）...")
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
    """只做尺寸限制，保持彩色原图。反色/CLAHE 会和在线 demo 不一致。"""
    h, w = crop.shape[:2]
    scale = 1.0
    if min(h, w) < MIN_CROP_SIDE:
        scale = MIN_CROP_SIDE / max(min(h, w), 1)
    if max(h, w) * scale > MAX_CROP_SIDE:
        scale = MAX_CROP_SIDE / max(h, w)
    if abs(scale - 1.0) > 0.05:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return crop


def run_ocr(ocr, crop: np.ndarray) -> str:
    if crop is None or crop.size == 0:
        return ""
    result = ocr.predict(
        prepare_crop(crop),
        text_det_thresh=DET_THRESH,
        text_det_box_thresh=DET_BOX_THRESH,
        text_det_unclip_ratio=DET_UNCLIP,
        text_rec_score_thresh=REC_SCORE_THRESH,
    )
    texts: list[str] = []
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
    return " ".join(texts)


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
    cap = open_camera(CAMERA_ID)
    print("摄像头已打开，按 Q 退出，按 S 保存当前帧")
    print(
        f"OCR=PP-OCRv6_tiny  无标签每 {OCR_RETRY} 帧重试  有标签每 {OCR_REFRESH} 帧复核  "
        f"det={DET_THRESH}/{DET_BOX_THRESH}  rec>={REC_SCORE_THRESH}"
    )

    frame_idx = 0
    ocr_cache: dict[int, OcrSlot] = {}
    last_seen: dict[int, int] = {}
    ocr_bind_mem: dict = {}
    next_tid = 1
    t0 = time.time()
    fps_show = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("读帧失败")
            break

        orig = frame.copy()
        result = model.predict(
            source=orig,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE,
            verbose=False,
        )[0]
        vis = np.ascontiguousarray(orig.copy())

        frame_idx += 1
        do_ocr = frame_idx % max(OCR_EVERY, 1) == 0
        det_labels: list[tuple[np.ndarray, int, str, float]] = []
        bottle_rows: list[tuple[int, dict, np.ndarray, float, float]] = []
        bottle_obs: list = []
        spatial: list[tuple[np.ndarray, float, dict]] = []
        for det in _iter_dets(result):
            name = class_names.get(det["cls"], str(det["cls"]))
            pts = det["pts"]
            conf = det["conf"]
            if name == OCR_CLASS:
                center = pts.mean(axis=0)
                x1, y1 = pts.min(axis=0)
                x2, y2 = pts.max(axis=0)
                size = float(max(x2 - x1, y2 - y1, 1.0))
                spatial.append((center, size, det))
            else:
                det_labels.append((pts, det["cls"], name, conf))

        tids, next_tid = assign_spatial_tids(
            [(c, s) for c, s, _d in spatial],
            ocr_cache,
            frame_idx,
            OCR_INHERIT_DIST,
            OCR_INHERIT_GAP,
            next_tid,
            OCR_STICKY_DIST,
        )
        for tid, (center, size, det) in zip(tids, spatial):
            last_seen[tid] = frame_idx
            pts = det["pts"]
            x1, y1 = pts.min(axis=0)
            x2, y2 = pts.max(axis=0)
            xyxy = np.array([float(x1), float(y1), float(x2), float(y2)], dtype=np.float32)
            bottle_obs.append((tid, center, size, xyxy))
            bottle_rows.append((tid, det, center, size, float(max(y2 - y1, 1.0))))

        for src, dst, lab in rebind_ocr_cache(
            ocr_cache,
            bottle_obs,
            frame_idx,
            last_seen,
            OCR_INHERIT_DIST,
            OCR_INHERIT_GAP,
            OCR_STICKY_DIST,
            ocr_bind_mem,
            OCR_HOLD_FRAMES,
        ):
            print(f"[OCR] 绑定 id={src} → {dst}  {bottle_label(lab)}")

        close_ids = crowded_bottle_tids(bottle_obs)
        blocked_ids = ocr_blocked_tids(ocr_bind_mem, frame_idx)
        active_ids = {tid for tid, _d, _c, _s, _h in bottle_rows}
        for tid, det, center, size, height in bottle_rows:
            record = ocr_cache.get(tid)
            pts = det["pts"]
            if (tid not in close_ids) and (tid not in blocked_ids) and should_run_ocr(record, frame_idx, do_ocr, OCR_RETRY, OCR_REFRESH):
                expand = neighbor_crop_expand(center, size, bottle_obs, tid, CROP_EXPAND)
                if det["kind"] == "obb":
                    crop = crop_obb(orig, pts, expand=expand)
                else:
                    crop = crop_aabb(orig, det["xyxy"], expand=expand)
                text = run_ocr(ocr, crop) if crop is not None else ""
                matched = match_reagent_label(text)
                old_lab = record.reagent if record is not None else ""
                new_rec = OcrSlot(
                    raw_text=matched,
                    reagent=matched,
                    center=center.copy(),
                    frame_id=frame_idx,
                    ocr_frame=frame_idx,
                    size=size,
                    height=height,
                )
                if record is not None:
                    new_rec.vel = np.asarray(record.vel, dtype=np.float32).copy()
                ocr_cache[tid] = merge_ocr_cache(record, new_rec)
                if matched and matched != old_lab:
                    print(f"[OCR] id={tid}  {text!r} -> {bottle_label(matched)}")
                elif text and not matched:
                    print(f"[OCR] id={tid}  {text!r} -> 未匹配，沿用{bottle_label(old_lab)}")
            name = bottle_label(ocr_cache.get(tid, OcrSlot()).raw_text)
            det_labels.append((pts, det["cls"], name, det["conf"]))

        expire_before = frame_idx - OCR_HOLD_FRAMES
        for cache_tid in list(ocr_cache):
            if cache_tid in active_ids:
                continue
            if last_seen.get(cache_tid, 0) <= expire_before:
                ocr_cache.pop(cache_tid, None)
                last_seen.pop(cache_tid, None)

        annotator = Annotator(vis, example="无菌水瓶")
        for pts, cls_id, name, conf in det_labels:
            annotator.box_label(pts, f"{name} {conf:.2f}", color=colors(cls_id, True))
        vis = np.ascontiguousarray(np.array(annotator.result(), copy=True))

        if frame_idx % 10 == 0:
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

        cv2.imshow("YOLO-OBB + OCR", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            out_dir = Path(r"E:\XMY\代码\ultralytics\runs\detect")
            out_dir.mkdir(parents=True, exist_ok=True)
            snap = out_dir / f"ocr_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(snap), vis)
            print(f"已保存 {snap}")
            # 把当前瓶子裁图另存，可直接丢到在线 demo 对比
            for j, det in enumerate(_iter_dets(result)):
                name = class_names.get(det["cls"], str(det["cls"]))
                if name != OCR_CLASS:
                    continue
                if det["kind"] == "obb":
                    crop = crop_obb(orig, det["pts"], expand=CROP_EXPAND)
                else:
                    crop = crop_aabb(orig, det["xyxy"], expand=CROP_EXPAND)
                if crop is None:
                    continue
                crop_path = out_dir / f"ocr_crop_{frame_idx:06d}_{j}.jpg"
                cv2.imwrite(str(crop_path), prepare_crop(crop))
                print(f"已保存裁图 {crop_path}  （把这张上传到 PP-OCRv6 在线 tiny 对比）")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
