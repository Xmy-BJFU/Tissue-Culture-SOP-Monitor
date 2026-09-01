"""手部关键点验证：YOLO 手套框 + RTMPose 21 点（预训练，不用标数据）。.

和 MediaPipe 的差别：RTMPose 看手的形状，不靠肤色找掌心，戴手套更有机会出点。
检测仍用已经能框出「手套」的 YOLO；RTMPose 只在框里估 21 点。

仍用 yolov26 环境，不要新建。首次运行会从 OpenMMLab 下载 ONNX。

    pip install rtmlib onnxruntime-gpu

用法:
    python hand_rtmpose.py
    python hand_rtmpose.py --source 0
    python hand_rtmpose.py --source "视频.mp4"

按键: Q 退出  S 保存  M 切换 ROI/整帧  空格暂停
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from hand_mediapipe import (  # 复用画骨架 / 中文 HUD / 裁框
    CONTEXT_CLASSES,
    DEFAULT_WEIGHTS,
    GLOVE_EXPAND,
    HAND_CLASS,
    JITTER_WINDOW,
    ROOT,
    STAT_WINDOW,
    TIP_IDS,
    draw_hand,
    draw_obb,
    expand_xyxy,
    open_source,
    parse_yolo,
    put_cn,
    tip_jitter_px,
)
from ultralytics import YOLO

try:
    from rtmlib import RTMDet, RTMPose
except ImportError as exc:
    raise SystemExit(
        "未安装 rtmlib。请先在 yolov26 环境执行:\n"
        "    pip install rtmlib onnxruntime-gpu\n"
        "若没有 CUDA，改成: pip install rtmlib onnxruntime"
    ) from exc

# OpenMMLab 手部预训练（5 个数据集，21 点，顺序与 MediaPipe 一致）
RTMDET_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmdet_nano_8xb32-300e_hand-267f9c8f.zip"
)
RTMPOSE_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip"
)
MIN_VALID_KPTS = 8


def yolo_device_to_rtm(device: str) -> str:
    d = str(device).lower()
    if d in {"cpu", "-1"}:
        return "cpu"
    if d.startswith("cuda"):
        return d if ":" in d else "cuda"
    return "cuda"


def enable_ort_cuda() -> None:
    """Onnxruntime-gpu 找不到系统 CUDA 时，借用 PyTorch 自带的 cublas/cudnn。."""
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
    """手套框优先；RTMDet 只补没有重叠的裸手。."""
    out = [list(b) for b in primary]
    for box in extra:
        if all(iou_xyxy(box, p) < iou_thr for p in out):
            out.append(list(box))
    return out


def glove_boxes_from_yolo(gloves, shape) -> tuple[list[list[float]], list[tuple[int, int, int, int]]]:
    boxes: list[list[float]] = []
    box_xy: list[tuple[int, int, int, int]] = []
    for _name, _conf, xyxy, _pts in gloves:
        x1, y1, x2, y2 = expand_xyxy(xyxy, shape, GLOVE_EXPAND)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([float(x1), float(y1), float(x2), float(y2)])
        box_xy.append((x1, y1, x2, y2))
    return boxes, box_xy


def match_hands_to_boxes(
    found: list[dict],
    box_xy: list[tuple[int, int, int, int]],
) -> tuple[list[dict], set[int]]:
    """手腕落在哪个手套框里，就算这只手套有点。框外的骨架（RTMDet 补的裸手）照样保留。."""
    used: set[int] = set()
    for hand in found:
        wx, wy, _ = hand["pts"][0]
        for i, (x1, y1, x2, y2) in enumerate(box_xy):
            if i in used:
                continue
            if x1 <= wx <= x2 and y1 <= wy <= y2:
                used.add(i)
                break
    return found, used


def pose_on_boxes(
    pose: RTMPose,
    bgr: np.ndarray,
    bboxes: list[list[float]],
    pose_conf: float,
) -> list[dict]:
    """RTMPose 对每个框都会出 21 点，必须靠分数滤掉瞎猜。."""
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
        hands.append({"pts": pts, "name": f"hand {mean_sc:.2f}", "score": mean_sc})
    hands.sort(key=lambda h: h["score"], reverse=True)
    return hands


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO 手套框 + RTMPose 21 点验证")
    p.add_argument("--source", default="0", help="摄像头编号或视频路径")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="YOLO-OBB 权重（找手套）")
    p.add_argument(
        "--mode",
        choices=("roi", "full"),
        default="roi",
        help="roi=只用 YOLO 手套框；full=YOLO 手套框 + RTMDet 补漏",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 手套置信度")
    p.add_argument("--pose-conf", type=float, default=0.25, help="关键点分数阈值，手套可略低")
    p.add_argument("--det-thr", type=float, default=0.25, help="整帧模式 RTMDet 阈值")
    p.add_argument("--device", default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    obb_path = Path(args.weights)
    if not obb_path.exists():
        raise FileNotFoundError(f"找不到 YOLO 权重: {obb_path}")

    rtm_dev = yolo_device_to_rtm(args.device)
    obb = YOLO(str(obb_path))
    print(f"YOLO 类别: {list(obb.names.values())}")
    if HAND_CLASS not in list(obb.names.values()):
        print(f"警告: 模型没有「{HAND_CLASS}」，手套框会空，只能靠 RTMDet 碰运气。")

    det, pose = make_rtm(rtm_dev, args.det_thr)
    cap, is_cam = open_source(args.source)
    mode = args.mode
    paused = False
    frame_id = 0
    t0 = time.time()
    fps_show = 0.0
    hit_hist: deque[int] = deque(maxlen=STAT_WINDOW)
    miss_hist: deque[int] = deque(maxlen=STAT_WINDOW)
    jitter_hist: deque[np.ndarray] = deque(maxlen=JITTER_WINDOW)
    snap_dir = ROOT / "runs" / "detect"
    snap_dir.mkdir(parents=True, exist_ok=True)

    print("戴手套看骨架。两种模式都在整幅画面上画点；差别只是找手的框从哪来。")
    print("ROI=只用 YOLO 手套框；整帧=YOLO 手套框为主，RTMDet 只补漏裸手。")
    print("按 Q 退出  S 保存  M 切换 ROI/整帧  空格暂停")

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
        dets = parse_yolo(
            obb.predict(source=orig, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        )
        gloves = [d for d in dets if d[0] == HAND_CLASS]
        vis = orig.copy()

        for name, conf, _xyxy, pts in dets:
            if name == HAND_CLASS:
                draw_obb(vis, pts, (0, 165, 255), f"{name} {conf:.2f}")
            elif name in CONTEXT_CLASSES:
                draw_obb(vis, pts, (180, 180, 180), f"{name} {conf:.2f}")

        roi_preview = None
        glove_miss = 0

        glove_boxes, box_xy = glove_boxes_from_yolo(gloves, orig.shape)
        glove_miss += max(len(gloves) - len(glove_boxes), 0)
        pose_boxes = list(glove_boxes)
        if mode == "full":
            extra = boxes_to_list(det(orig))
            pose_boxes = merge_boxes(glove_boxes, extra)
            for x1, y1, x2, y2 in extra:
                if all(iou_xyxy([x1, y1, x2, y2], g) < 0.45 for g in glove_boxes):
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (80, 255, 80), 1)

        for x1, y1, x2, y2 in box_xy:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 1)
            if roi_preview is None:
                roi_preview = orig[y1:y2, x1:x2]

        found = pose_on_boxes(pose, orig, pose_boxes, args.pose_conf)
        hands, used = match_hands_to_boxes(found, box_xy)
        glove_miss += max(len(glove_boxes) - len(used), 0)
        for i, (x1, y1, x2, y2) in enumerate(box_xy):
            if i not in used:
                cv2.putText(vis, "MISS", (x1 + 6, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        for hand in hands:
            draw_hand(vis, hand["pts"], hand["name"])

        n_tips = 0
        if hands:
            pts = hands[0]["pts"]
            n_tips = sum(1 for i in TIP_IDS if i < len(pts) and pts[i][2] >= args.pose_conf)
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

        if roi_preview is not None and roi_preview.size:
            thumb = cv2.resize(roi_preview, (160, 160))
            vis[8:168, vis.shape[1] - 168 : vis.shape[1] - 8] = thumb
            cv2.putText(vis, "ROI", (vis.shape[1] - 160, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        hud = [
            f"RTMPose  {'仅YOLO手套框' if mode == 'roi' else 'YOLO手套+RTMDet补漏'}   FPS {fps_show:.1f}",
            f"手套框 {len(gloves)}   姿态手 {len(hands)}   指尖 {n_tips}/5",
            f"近{len(hit_hist)}帧检出率 {hit_rate:.0f}%   手套漏检 {miss_rate:.0f}%",
            f"食指抖动 {jitter:.1f} px",
            "Q退出  S保存  M切换  空格暂停",
        ]
        if glove_miss:
            hud.insert(2, f"本帧漏检 {glove_miss} 只手套")
        vis = put_cn(vis, hud)
        cv2.imshow("Hand RTMPose", vis)
        key = cv2.waitKey(0 if paused else 1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("s"), ord("S")):
            snap = snap_dir / f"hand_rtm_{frame_id:06d}.jpg"
            cv2.imwrite(str(snap), vis)
            print(f"已保存 {snap}")
        if key in (ord("m"), ord("M")):
            mode = "full" if mode == "roi" else "roi"
            jitter_hist.clear()
            print(f"切换到 {'整帧(YOLO手套+RTMDet补漏)' if mode == 'full' else '仅YOLO手套框'}")
        if key == 32:
            paused = not paused

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
