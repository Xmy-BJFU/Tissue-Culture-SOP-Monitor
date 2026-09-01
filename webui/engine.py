"""后台推理会话：复用 video_track_ocr_hand_pose 的检测与动作规则。."""

from __future__ import annotations

import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import video_track_ocr_hand_pose as core

SOAK_SEC = {"酒精": 30, "无菌水": 30, "次氯酸钠": 15 * 60}
EXPECTED_ROLES = ("酒精", "无菌水", "次氯酸钠", "灭菌瓶", "培养基")
SOP_STEPS = [
    ("ppe", "防护检查", "手套等防护用品"),
    ("sterile", "工具灭菌", "镊子 + 美工刀插入灭菌器孔区"),
    ("alcohol", "酒精浸泡", "倒完回正起算 30s±10%，倒出液体停表"),
    ("water1", "无菌水冲洗", "倒完回正起算 30s±10%，倒出液体停表"),
    ("naocl", "次氯酸钠消毒", "15 分钟±10% + 浸泡中可摇晃，倒出液体停表"),
    ("rinse", "无菌水冲洗×3", "倒出轮次累计"),
    ("cut", "切割种球", "手持美工刀往复切割"),
    ("insert", "斜插接种", "鳞茎斜插入培养基"),
]
STREAM_MAX_SIDE = 1920
JPEG_QUALITY = 93


class LabEngine:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.ready = False
        self.loading = False
        self.load_error = ""
        self.running = False
        self.paused = False
        self.thread: threading.Thread | None = None
        self.cap: cv2.VideoCapture | None = None

        self.weights = str(core.DEFAULT_WEIGHTS)
        self.device = "0"
        self.imgsz = 640
        self.conf = 0.25
        self.iou = 0.7
        self.pose_conf = 0.25
        self.det_thr = 0.25
        self.hand_mode = "glove"
        self.source = "0"
        self.overlay = {"boxes": True, "skeleton": True, "hole": True, "labels": True}

        self.model = None
        self.class_names: dict[int, str] = {}
        self.ocr = None
        self.rtm_det = None
        self.pose = None
        self.tracker = None
        self.action = None
        self.ocr_cache: dict = {}
        self.last_seen: dict = {}
        self.ocr_bind_mem: dict = {}

        self.latest_jpeg = _placeholder_jpeg("等待开始实训")
        self.fps = 0.0
        self.frame_id = 0
        self.session_id = ""
        self.student = ""
        self.session_started_at = 0.0
        self.elapsed_hold = 0.0
        self.events: deque = deque(maxlen=400)
        self.objects: list[dict] = []
        self.hud_lines: list[str] = []
        self.holding: list[str] = []
        self.fired_flags: dict = {}
        self.action_trace: list = []
        self.ppe = {"手套": False, "口罩": False, "帽子": False}
        self.soaks: dict = {}
        self.alerts: list[dict] = []
        self._reset_session_logic()

    def _reset_session_logic(self) -> None:
        self.soaks = {
            name: {
                "target": sec,
                "started_at": None,
                "elapsed": 0.0,
                "running": False,
                "done": False,
                "early": False,
            }
            for name, sec in SOAK_SEC.items()
        }
        self.rinse_count = 0
        self.ppe_seen = False
        self.alerts = []

    def thresholds(self) -> dict:
        return {
            "pour_tilt_deg": core.POUR_TILT_DEG,
            "pour_upright_deg": core.POUR_UPRIGHT_DEG,
            "cut_span_px": core.CUT_SPAN_PX,
            "cut_hold_frames": core.CUT_HOLD_FRAMES,
            "shake_min_range": core.SHAKE_MIN_RANGE,
            "insert_angle_ok": list(core.INSERT_ANGLE_OK),
        }

    def apply_thresholds(self, data: dict) -> None:
        if "pour_tilt_deg" in data:
            core.POUR_TILT_DEG = float(data["pour_tilt_deg"])
        if "pour_upright_deg" in data:
            core.POUR_UPRIGHT_DEG = float(data["pour_upright_deg"])
        if "cut_span_px" in data:
            core.CUT_SPAN_PX = float(data["cut_span_px"])
        if "cut_hold_frames" in data:
            core.CUT_HOLD_FRAMES = int(data["cut_hold_frames"])
        if "shake_min_range" in data:
            core.SHAKE_MIN_RANGE = float(data["shake_min_range"])
        if "insert_angle_ok" in data:
            lo, hi = data["insert_angle_ok"]
            core.INSERT_ANGLE_OK = (float(lo), float(hi))

    def load_models(self) -> None:
        with self.lock:
            if self.ready or self.loading:
                return
            self.loading = True
            self.load_error = ""
        try:
            weights = Path(self.weights)
            if not weights.exists():
                raise FileNotFoundError(f"找不到权重: {weights}")
            model = YOLO(str(weights))
            ocr = core.load_ocr()
            rtm_dev = core.yolo_device_to_rtm(self.device)
            rtm_det, pose = core.make_rtm(rtm_dev, self.det_thr)
            with self.lock:
                self.model = model
                self.class_names = model.names
                self.ocr = ocr
                self.rtm_det = rtm_det
                self.pose = pose
                self.ready = True
                self.loading = False
        except Exception as exc:
            with self.lock:
                self.loading = False
                self.load_error = str(exc)
                self.ready = False
            raise

    def start(self, source: str | None = None, student: str = "") -> None:
        if not self.ready:
            self.load_models()
        self.stop()
        if source is not None:
            self.source = str(source)
        self.student = student or self.student
        cap, _is_cam = core.open_source(self.source)
        tracker = core.load_bytetrack(core.TRACK_BUFFER)
        tracker.args.match_thresh = core.TRACK_MATCH_THRESH
        with self.lock:
            self.cap = cap
            self.tracker = tracker
            self.action = core.ActionEngine()
            self.ocr_cache = {}
            self.last_seen = {}
            self.ocr_bind_mem = {}
            self.frame_id = 0
            self.session_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            self.session_started_at = time.time()
            self.elapsed_hold = 0.0
            self.events.clear()
            self._reset_session_logic()
            self.running = True
            self.paused = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        self._push_event("系统", f"实训开始  source={self.source}  手模式={self.hand_mode}")

    def stop(self) -> None:
        self.running = False
        if self.session_started_at:
            self.elapsed_hold = time.time() - self.session_started_at
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def pause(self, value: bool | None = None) -> None:
        self.paused = (not self.paused) if value is None else bool(value)
        if self.paused and self.action is not None:
            self.action.note_pause()

    def reset_actions(self) -> None:
        if self.action is not None:
            self.action.reset()
        self._reset_session_logic()
        self._push_event("系统", "已重置动作与计时状态")

    def snapshot(self) -> str | None:
        snap_dir = ROOT / "runs" / "detect"
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / f"web_{self.frame_id:06d}.jpg"
        with self.lock:
            jpeg = self.latest_jpeg
        if not jpeg:
            return None
        path.write_bytes(jpeg)
        self._push_event("系统", f"已截图 {path.name}")
        return str(path)

    def _push_event(self, kind: str, message: str, level: str = "info") -> None:
        item = {
            "ts": time.time(),
            "clock": datetime.now().strftime("%H:%M:%S"),
            "kind": kind,
            "message": message,
            "level": level,
            "frame": self.frame_id,
        }
        with self.lock:
            self.events.appendleft(item)

    def _loop(self) -> None:
        last_fps_t = time.time()
        fps_n = 0
        while self.running and self.cap is not None:
            if self.paused:
                time.sleep(0.03)
                continue
            ok, frame = self.cap.read()
            if not ok:
                if str(self.source).isdigit():
                    self._push_event("系统", "读帧失败，已停止", "warn")
                    break
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            try:
                vis, objects, view = self._process(frame)
            except Exception as exc:
                self._push_event("系统", f"推理异常: {exc}", "danger")
                time.sleep(0.2)
                continue
            h, w = vis.shape[:2]
            side = max(h, w)
            if side > STREAM_MAX_SIDE:
                scale = STREAM_MAX_SIDE / side
                vis = cv2.resize(vis, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            fps_n += 1
            now = time.time()
            if now - last_fps_t >= 0.8:
                self.fps = fps_n / (now - last_fps_t)
                fps_n = 0
                last_fps_t = now
            with self.lock:
                if ok:
                    self.latest_jpeg = buf.tobytes()
                self.objects = objects
                if view is not None:
                    self.hud_lines = list(view.hud_lines)
                    self.action_trace = list(view.trace or [])
            if self.action is not None:
                self.soaks = self.action.soak_state()
        self.running = False

    def _process(self, orig: np.ndarray):
        self.frame_id += 1
        result = self.model.predict(
            source=orig,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]
        dets = core.parse_detections(result)
        active, _removed = core.update_tracks(self.tracker, dets, orig)
        active_ids = {tid for tid, _, _ in active}
        do_ocr = self.frame_id % max(core.OCR_EVERY, 1) == 0
        gloves = [
            (self.class_names.get(d.cls_id, str(d.cls_id)), d.conf, d.xyxy, d.pts)
            for d in dets
            if self.class_names.get(d.cls_id, str(d.cls_id)) == core.HAND_CLASS
        ]
        tracked_glove_xyxy: list[list[float]] = []
        items: list[core.TrackedItem] = []
        box_rows: list[tuple] = []
        bottle_obs: list[tuple[int, np.ndarray, float]] = []
        for tid, det, _track_score in active:
            self.last_seen[tid] = self.frame_id
            if self.class_names.get(det.cls_id, str(det.cls_id)) == core.OCR_CLASS:
                bottle_obs.append(
                    (
                        tid,
                        det.pts.mean(axis=0),
                        float(max(det.xywh[2], det.xywh[3])),
                        np.asarray(det.xyxy, dtype=np.float32),
                    )
                )
        for src, dst, lab in core.rebind_ocr_cache(
            self.ocr_cache,
            bottle_obs,
            self.frame_id,
            self.last_seen,
            core.OCR_INHERIT_DIST,
            core.OCR_INHERIT_GAP,
            core.OCR_STICKY_DIST,
            self.ocr_bind_mem,
            core.OCR_HOLD_FRAMES,
        ):
            self._push_event("OCR", f"绑定 id={src} → {dst}  {core.bottle_label(lab)}")
        close_ids = core.crowded_bottle_tids(bottle_obs)
        blocked_ids = core.ocr_blocked_tids(self.ocr_bind_mem, self.frame_id)

        for tid, det, track_score in active:
            cls_name = self.class_names.get(det.cls_id, str(det.cls_id))
            center = det.pts.mean(axis=0)
            name = cls_name
            role = ""
            size = float(max(det.xywh[2], det.xywh[3]))
            if cls_name == core.HAND_CLASS:
                tracked_glove_xyxy.append([float(v) for v in det.xyxy])
            if cls_name == core.OCR_CLASS:
                record = self.ocr_cache.get(tid)
                if (
                    (tid not in close_ids)
                    and (tid not in blocked_ids)
                    and core.should_run_ocr(record, self.frame_id, do_ocr, core.OCR_RETRY, core.OCR_REFRESH)
                ):
                    expand = core.neighbor_crop_expand(center, size, bottle_obs, tid, core.CROP_EXPAND)
                    crop = (
                        core.crop_obb(orig, det.pts, expand=expand)
                        if det.pts is not None and det.pts.shape == (4, 2)
                        else core.crop_aabb(orig, det.xyxy, expand=expand)
                    )
                    text, score = core.run_ocr(self.ocr, crop) if crop is not None else ("", 0.0)
                    matched = core.match_reagent_label(text)
                    if core.neighbor_label_conflict(tid, matched, self.ocr_cache, bottle_obs):
                        matched = ""
                    old_lab = record.reagent if record is not None else ""
                    new_rec = core.OcrRecord(
                        raw_text=matched,
                        score=score,
                        reagent=matched,
                        center=center.copy(),
                        frame_id=self.frame_id,
                        ocr_frame=self.frame_id,
                        size=size,
                        height=float(max(det.xyxy[3] - det.xyxy[1], 1.0)),
                    )
                    if record is not None:
                        new_rec.vel = np.asarray(record.vel, dtype=np.float32).copy()
                    self.ocr_cache[tid] = core.merge_ocr_cache(record, new_rec)
                    if matched and matched != old_lab:
                        self._push_event("OCR", f"id={tid}  {text} → {core.bottle_label(matched)}")
                rec_now = self.ocr_cache.get(tid, core.OcrRecord())
                role = rec_now.reagent
                name = core.bottle_label(rec_now.raw_text)
            items.append(
                core.TrackedItem(
                    tid=tid,
                    cls_name=cls_name,
                    role=role,
                    name=name,
                    det=det,
                    pts=det.pts,
                    xyxy=det.xyxy,
                    center=center.astype(np.float32),
                    tilt=core.obb_tilt_from_vertical(det.pts),
                )
            )
            box_rows.append((det.pts, det.cls_id, name, track_score, tid))

        expire_before = self.frame_id - core.OCR_HOLD_FRAMES
        for cache_tid in list(self.ocr_cache):
            if cache_tid in active_ids:
                continue
            if self.last_seen.get(cache_tid, 0) <= expire_before:
                self.ocr_cache.pop(cache_tid, None)
                self.last_seen.pop(cache_tid, None)

        pose_boxes = (
            core.glove_boxes_from_yolo(gloves, orig.shape)
            if self.hand_mode == "glove"
            else core.boxes_to_list(self.rtm_det(orig))
        )
        hands = core.pose_on_boxes(self.pose, orig, pose_boxes, self.pose_conf)
        view = self.action.update(self.frame_id, items, hands, self.class_names)
        for msg in view.fired:
            self._on_action(msg)

        names_now = {it.cls_name for it in items}
        self.ppe = {
            "手套": core.HAND_CLASS in names_now or bool(gloves),
            "口罩": "口罩" in names_now,
            "帽子": "帽子" in names_now,
        }
        if self.ppe["手套"]:
            self.ppe_seen = True

        vis = self._draw(orig, box_rows, gloves, tracked_glove_xyxy, hands, view)
        objects = [
            {
                "tid": it.tid,
                "cls": it.cls_name,
                "role": it.role,
                "name": it.name,
                "conf": round(float(it.det.conf), 3),
                "tilt": round(float(it.tilt), 1),
                "holding": it.tid in view.held_tids,
            }
            for it in items
        ]
        self.holding = [it.name for it in items if it.tid in view.held_tids]
        self.fired_flags = {
            "tweezers_ok": self.action.tweezers_ok,
            "knife_ok": self.action.knife_ok,
            "step1_ok": self.action.step1_ok,
            "naocl_poured": self.action.naocl_poured,
            "last_pour": self.action.last_pour,
            "pour_done_n": dict(self.action.pour_done_n),
            "dump_n": self.action.dump_n,
            "shake_scored": self.action.shake_scored,
            "shake_unscored": self.action.shake_unscored,
            "cutting_ok": self.action.cutting_ok,
            "insert_ok": self.action.insert_ok,
            "insert_angle_ok": self.action.insert_angle_ok,
            "live_pour": self.action.live_pour,
            "live_dump": self.action.live_dump,
            "live_shake": self.action.live_shake,
        }
        self.action_trace = list(view.trace or [])
        self.soaks = self.action.soak_state()
        return vis, objects, view

    def _on_action(self, msg: str) -> None:
        level = "ok"
        kind = "计时" if msg.startswith("计时") else "动作"
        if "不计分" in msg or "偏早" in msg or "超时" in msg:
            level = "warn"
        if "合格" in msg and "计时" in msg:
            level = "ok"
        if "倒出" in msg and self.action and self.action.naocl_poured:
            self.rinse_count += 1
        self._push_event(kind, msg, level)

    def _start_soak(self, name: str) -> None:
        for other, soak in self.soaks.items():
            if other != name and soak["running"]:
                soak["running"] = False
                soak["done"] = True
        soak = self.soaks[name]
        soak["started_at"] = time.time()
        soak["elapsed"] = 0.0
        soak["running"] = True
        soak["done"] = False
        soak["early"] = False
        self._push_event("计时", f"{name}浸泡计时开始（{SOAK_SEC[name]}s）", "ok")

    def _tick_soaks(self) -> None:
        now = time.time()
        for name, soak in self.soaks.items():
            if soak["running"] and soak["started_at"]:
                soak["elapsed"] = now - soak["started_at"]
                if soak["elapsed"] >= soak["target"]:
                    soak["running"] = False
                    soak["done"] = True
                    soak["elapsed"] = soak["target"]
                    self._push_event("计时", f"{name}浸泡时长达标", "ok")

    def _draw(self, orig, box_rows, gloves, tracked_glove_xyxy, hands, view):
        vis = np.ascontiguousarray(orig.copy())
        annotator = Annotator(vis, example="无菌水瓶灭菌摇晃切割斜插")
        ov = self.overlay
        if ov.get("boxes", True):
            for pts, cls_id, name, track_score, tid in box_rows:
                holding = tid in view.held_tids
                name if ov.get("labels", True) else ""
                if ov.get("labels", True):
                    tag = f"持 {name} {track_score:.2f}" if holding else f"{name} {track_score:.2f}"
                else:
                    tag = ""
                annotator.box_label(pts, tag, color=colors(cls_id, True))
            if self.hand_mode == "glove":
                glove_cls = next((i for i, n in self.class_names.items() if n == core.HAND_CLASS), 0)
                for _name, conf, xyxy, pts in gloves:
                    xy = [float(v) for v in xyxy]
                    if any(core.iou_xyxy(xy, t) >= 0.3 for t in tracked_glove_xyxy):
                        continue
                    annotator.box_label(pts, f"{core.HAND_CLASS} {conf:.2f}", color=colors(glove_cls, True))
        vis = np.ascontiguousarray(np.array(annotator.result(), copy=True))
        if ov.get("hole", True) and view.hole_xyxy is not None:
            x1, y1, x2, y2 = [int(v) for v in view.hole_xyxy]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 1, cv2.LINE_AA)
        if ov.get("skeleton", True):
            for hand in hands:
                core.draw_hand(vis, hand["pts"])
        return vis

    def sop_status(self) -> list[dict]:
        flags = self.fired_flags
        alcohol = self.soaks["酒精"]
        water = self.soaks["无菌水"]
        naocl = self.soaks["次氯酸钠"]
        states = {
            "ppe": "done" if self.ppe_seen else ("active" if self.running else "idle"),
            "sterile": "done"
            if flags.get("step1_ok")
            else ("active" if flags.get("tweezers_ok") or flags.get("knife_ok") else "idle"),
            "alcohol": "done" if alcohol["done"] else ("active" if alcohol["running"] else "idle"),
            "water1": "done" if water["done"] else ("active" if water["running"] else "idle"),
            "naocl": "done"
            if (naocl["done"] and flags.get("shake_scored"))
            else ("active" if naocl["running"] or flags.get("naocl_poured") else "idle"),
            "rinse": "done" if self.rinse_count >= 3 else ("active" if self.rinse_count else "idle"),
            "cut": "done" if flags.get("cutting_ok") else "idle",
            "insert": "done" if flags.get("insert_ok") else "idle",
        }
        out = []
        for key, title, hint in SOP_STEPS:
            out.append({"id": key, "title": title, "hint": hint, "state": states[key]})
        return out

    def score(self) -> dict:
        sop = self.sop_status()
        done = sum(1 for s in sop if s["state"] == "done")
        step_ratio = done / max(len(sop), 1)
        roles = {obj["role"] for obj in self.objects if obj.get("role")}
        item_ratio = len(roles & set(EXPECTED_ROLES)) / len(EXPECTED_ROLES)
        time_hits = []
        for soak in self.soaks.values():
            if soak.get("done") or soak.get("running"):
                verdict = soak.get("verdict") or ""
                if verdict == "ok":
                    time_hits.append(1.0)
                elif verdict in ("early", "late") or soak.get("early"):
                    time_hits.append(0.4)
                else:
                    time_hits.append(0.8 if soak.get("running") else 1.0)
        time_ratio = float(np.mean(time_hits)) if time_hits else 0.0
        safety = 1.0 if self.ppe_seen else 0.0
        step_s = round(40 * step_ratio, 1)
        item_s = round(30 * item_ratio, 1)
        time_s = round(10 * time_ratio, 1)
        safe_s = round(20 * safety, 1)
        total = round(step_s + item_s + time_s + safe_s, 1)
        return {
            "total": total,
            "dims": [
                {
                    "id": "step",
                    "name": "实训步骤准确率",
                    "weight": 40,
                    "score": step_s,
                    "detail": f"{done}/{len(sop)} 步完成",
                },
                {
                    "id": "item",
                    "name": "操作物品正确率",
                    "weight": 30,
                    "score": item_s,
                    "detail": f"已识别 {len(roles & set(EXPECTED_ROLES))}/{len(EXPECTED_ROLES)} 类标签",
                },
                {
                    "id": "time",
                    "name": "操作时间规范性",
                    "weight": 10,
                    "score": time_s,
                    "detail": "倒完回正起算，倒出液体停表，±10% 裕量；偏早或超时扣分",
                },
                {
                    "id": "safety",
                    "name": "操作安全与规范",
                    "weight": 20,
                    "score": safe_s,
                    "detail": "已检出手套" if self.ppe_seen else "未检出手套",
                },
            ],
        }

    def state(self) -> dict:
        if self.running and self.session_started_at:
            elapsed = time.time() - self.session_started_at
        else:
            elapsed = self.elapsed_hold
        soaks = {}
        for name, soak in self.soaks.items():
            soaks[name] = {
                **soak,
                "remain": max(0.0, soak["target"] - soak["elapsed"]),
                "progress": min(1.0, soak["elapsed"] / soak["target"]) if soak["target"] else 0,
            }
        return {
            "ready": self.ready,
            "loading": self.loading,
            "load_error": self.load_error,
            "running": self.running,
            "paused": self.paused,
            "fps": round(self.fps, 1),
            "frame_id": self.frame_id,
            "hand_mode": self.hand_mode,
            "source": self.source,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "weights": self.weights,
            "classes": list(self.class_names.values()) if self.class_names else [],
            "session_id": self.session_id,
            "student": self.student,
            "elapsed": round(elapsed, 1),
            "overlay": self.overlay,
            "ppe": self.ppe,
            "holding": self.holding,
            "objects": self.objects,
            "hud": self.hud_lines,
            "flags": self.fired_flags,
            "action_trace": self.action_trace,
            "soaks": soaks,
            "rinse_count": self.rinse_count,
            "sop": self.sop_status(),
            "score": self.score(),
            "thresholds": self.thresholds(),
            "events": list(self.events)[:80],
            "alerts": self.alerts,
        }

    def report(self) -> dict:
        st = self.state()
        return {
            "title": "朱顶红无菌体系建立 · 实训报告",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": self.session_id,
            "student": self.student or "未填写",
            "source": self.source,
            "elapsed": st["elapsed"],
            "score": st["score"],
            "sop": st["sop"],
            "flags": st["flags"],
            "soaks": st["soaks"],
            "ppe": st["ppe"],
            "events": list(self.events),
        }


def _placeholder_jpeg(text: str) -> bytes:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (12, 14, 20)
    for y in range(0, 720, 40):
        cv2.line(img, (0, y), (1280, y), (22, 28, 38), 1)
    for x in range(0, 1280, 40):
        cv2.line(img, (x, 0), (x, 720), (22, 28, 38), 1)
    cv2.putText(img, "STERILE LAB OS", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 180), 2, cv2.LINE_AA)
    cv2.putText(img, text, (80, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (180, 190, 200), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else b""


engine = LabEngine()
