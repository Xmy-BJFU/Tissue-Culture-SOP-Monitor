"""瓶子 OCR 缓存与跟踪换 ID 时的标签绑定。.

规则：
- 未识别到标签：隔若干帧再检
- 已识别到标签：仍按间隔复核；新结果不同则改用新标签；这次没检到字则沿用缓存
- 快移：ByteTrack 换 ID 后按预测位置唯一匹配；速度大时暂缓 OCR，避免糊图冲掉缓存
- 两瓶过近：分不清则不改绑，避免串标；裁图缩小以免读到邻瓶文字
- 遮挡抢标（方案 A）：前面/较矮的瓶不能继承后面已锁定的 OCR；交叉期间不做 OCR；
  分开后标签回到高度更接近原瓶的那只
- 拿起丢框：仅当上一帧至少两瓶重叠、一只框消失、留下的框明显是另一只时才停放；
  单瓶移动或跟踪换 ID 时标签必须跟着走
"""

from __future__ import annotations

import numpy as np

REAGENT_LABELS = ("酒精", "无菌水", "次氯酸钠", "灭菌瓶", "培养基")
REAGENT_ALIASES = {
    "酒精": ("酒精", "乙醇", "alcohol", "etoh"),
    "无菌水": ("无菌水", "无菌水瓶", "灭菌水", "蒸馏水"),
    "次氯酸钠": ("次氯酸钠", "次氯酸", "次氯", "84"),
    "灭菌瓶": ("灭菌瓶", "灭菌罐"),
    "培养基": ("培养基", "培养皿"),
}
REAGENT_CHAR_WEIGHTS = {
    "酒精": {"酒": 3.0, "精": 3.0, "乙": 2.0, "醇": 2.0},
    "无菌水": {"无": 2.5, "水": 4.0, "蒸": 2.0, "馏": 2.0, "菌": 0.6},
    "次氯酸钠": {"氯": 3.0, "钠": 3.0, "次": 2.0, "酸": 1.5},
    "灭菌瓶": {"灭": 2.5, "瓶": 4.0, "罐": 3.5, "菌": 0.6},
    "培养基": {"培": 3.0, "养": 3.0, "基": 2.5},
}


def normalize_ocr_text(ocr_text: str) -> str:
    text = "".join((ocr_text or "").split())
    return text.replace("茵", "菌").replace("滅", "灭")


def match_water_or_sterile(text: str) -> str:
    """无菌水 vs 灭菌瓶：用水 / 瓶 / 罐拍板。.

    共有字「菌」和短词「无菌」「灭菌」不再当结论——OCR 常把「无/灭」读反， 「灭菌水」应归无菌水，「无菌瓶」（灭读成无、没有水）应归灭菌瓶。
    """
    if any(p in text for p in ("无菌水", "灭菌水", "蒸馏水", "无菌水瓶")):
        return "无菌水"
    if any(p in text for p in ("灭菌瓶", "灭菌罐")):
        return "灭菌瓶"
    has_water = any(ch in text for ch in "水蒸馏")
    has_vessel = any(ch in text for ch in "瓶罐")
    has_wu = "无" in text
    has_mie = "灭" in text
    if has_water:
        return "无菌水"
    if has_vessel and (has_wu or has_mie):
        return "灭菌瓶"
    return ""


def match_reagent_label(ocr_text: str) -> str:
    text = normalize_ocr_text(ocr_text)
    if not text:
        return ""
    pair = match_water_or_sterile(text)
    hay = text.casefold()
    alias_hits: list[tuple[int, str]] = []
    for label, aliases in REAGENT_ALIASES.items():
        if label in ("无菌水", "灭菌瓶"):
            continue
        for alias in (label, *aliases):
            if alias and alias.casefold() in hay:
                alias_hits.append((len(alias), label))
    if alias_hits:
        alias_hits.sort(key=lambda x: x[0], reverse=True)
        return alias_hits[0][1]
    if pair:
        return pair
    scored: list[tuple[float, float, str]] = []
    for label in REAGENT_LABELS:
        if label in ("无菌水", "灭菌瓶"):
            continue
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


def predicted_center(rec, frame_id: int) -> np.ndarray:
    dt = int(max(0, frame_id - int(getattr(rec, "frame_id", frame_id))))
    dt = min(dt, 12)
    vel = np.asarray(getattr(rec, "vel", np.zeros(2)), dtype=np.float32).reshape(-1)[:2]
    center = np.asarray(rec.center, dtype=np.float32).reshape(-1)[:2]
    return center + vel * float(dt)


def record_speed(rec) -> float:
    if rec is None:
        return 0.0
    vel = np.asarray(getattr(rec, "vel", np.zeros(2)), dtype=np.float32).reshape(-1)[:2]
    return float(np.linalg.norm(vel))


def adaptive_max_dist(rec, base: float) -> float:
    speed = record_speed(rec)
    size = float(getattr(rec, "size", 0.0) or 0.0)
    return float(base + min(160.0, speed * 8.0) + 0.35 * size)


def update_ocr_motion(rec, center: np.ndarray, size: float, frame_id: int, height: float | None = None) -> None:
    center = np.asarray(center, dtype=np.float32).reshape(-1)[:2]
    old = np.asarray(rec.center, dtype=np.float32).reshape(-1)[:2]
    if int(getattr(rec, "frame_id", 0)) > 0:
        dt = max(1, int(frame_id - rec.frame_id))
        inst = (center - old) / float(dt)
        vel = np.asarray(getattr(rec, "vel", np.zeros(2)), dtype=np.float32).reshape(-1)[:2]
        vel = 0.65 * vel + 0.35 * inst
        speed = float(np.linalg.norm(vel))
        if speed > 90.0:
            vel = vel * (90.0 / speed)
        rec.vel = vel.astype(np.float32)
    else:
        rec.vel = np.zeros(2, dtype=np.float32)
    rec.center = center.copy()
    rec.size = float(size)
    rec.frame_id = int(frame_id)
    if height is not None and float(height) > 1:
        rec.height = float(height)


def should_run_ocr(
    record,
    frame_id: int,
    do_ocr: bool,
    retry: int,
    refresh: int,
    speed_skip: float = 32.0,
) -> bool:
    if not do_ocr:
        return False
    if record is None:
        return True
    has_label = bool(getattr(record, "reagent", None))
    if has_label and record_speed(record) >= speed_skip:
        return False
    age = int(frame_id - int(getattr(record, "ocr_frame", getattr(record, "frame_id", 0))))
    if not has_label:
        return age >= max(1, retry)
    return age >= max(1, refresh)


def rec_height(rec) -> float:
    return float(getattr(rec, "height", 0.0) or getattr(rec, "size", 0.0) or 0.0)


def merge_ocr_cache(old, new):
    if old is None:
        return new
    if getattr(new, "reagent", ""):
        new.vel = np.asarray(getattr(old, "vel", np.zeros(2)), dtype=np.float32).copy()
        if not getattr(new, "size", 0):
            new.size = float(getattr(old, "size", 0.0) or 0.0)
        if not getattr(new, "height", 0) and getattr(old, "height", 0):
            new.height = float(old.height)
        return new
    old.center = np.asarray(new.center, dtype=np.float32).reshape(-1)[:2].copy()
    old.frame_id = int(new.frame_id)
    old.ocr_frame = int(getattr(new, "ocr_frame", new.frame_id))
    if getattr(new, "raw_text", "") and not getattr(old, "reagent", ""):
        old.raw_text = str(new.raw_text)
        old.score = float(getattr(new, "score", 0.0) or 0.0)
    if getattr(new, "size", 0):
        old.size = float(new.size)
    vel = getattr(new, "vel", None)
    if vel is not None:
        old.vel = np.asarray(vel, dtype=np.float32).reshape(-1)[:2].copy()
    return old


def neighbor_crop_expand(
    center: np.ndarray,
    size: float,
    bottles: list,
    tid: int,
    base_expand: float,
) -> float:
    """两瓶过近时缩小裁图，避免把邻瓶文字读进来。."""
    center = np.asarray(center, dtype=np.float32).reshape(-1)[:2]
    min_d = 1e9
    for b in bottles:
        other_tid, other_c, other_s, _x, _h = unpack_bottle(b)
        if int(other_tid) == int(tid):
            continue
        d = float(np.linalg.norm(other_c - center))
        min_d = min(min_d, d)
        limit = 0.7 * (float(size) + float(other_s))
        if d < max(limit, 48.0):
            return 1.05
    if min_d < float(size) * 1.35:
        return min(base_expand, 1.04)
    return base_expand


def neighbor_label_conflict(
    tid: int,
    matched: str,
    ocr_cache: dict,
    bottles: list,
) -> bool:
    """标签不唯一：多只瓶子可以同为酒精/无菌水等。近邻读串靠缩小裁图处理，不再互斥。."""
    return False


def unpack_bottle(b) -> tuple[int, np.ndarray, float, np.ndarray | None, float]:
    tid = int(b[0])
    center = np.asarray(b[1], dtype=np.float32).reshape(-1)[:2]
    size = float(b[2])
    xyxy = None
    if len(b) > 3 and b[3] is not None:
        xyxy = np.asarray(b[3], dtype=np.float32).reshape(-1)[:4]
        height = float(max(xyxy[3] - xyxy[1], 1.0))
    else:
        height = size
    return tid, center, size, xyxy, height


def _xyxy_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area = max(ax2 - ax1, 1.0) * max(ay2 - ay1, 1.0) + max(bx2 - bx1, 1.0) * max(by2 - by1, 1.0) - inter
    return float(inter / max(area, 1.0))


def bottles_overlap(b1, b2) -> bool:
    _t1, c1, s1, x1, _h1 = unpack_bottle(b1)
    _t2, c2, s2, x2, _h2 = unpack_bottle(b2)
    d = float(np.linalg.norm(c1 - c2))
    if x1 is not None and x2 is not None:
        if _xyxy_iou(x1, x2) >= 0.08:
            return True
        x_ov = min(float(x1[2]), float(x2[2])) - max(float(x1[0]), float(x2[0]))
        y_ov = min(float(x1[3]), float(x2[3])) - max(float(x1[1]), float(x2[1]))
        min_w = min(float(x1[2] - x1[0]), float(x2[2] - x2[0]))
        if x_ov > 0.32 * max(min_w, 1.0) and y_ov > 0 and d < 0.95 * 0.5 * (s1 + s2) + 28:
            return True
    return d < 0.78 * 0.5 * (s1 + s2) + 22.0


def front_tid(b1, b2) -> int:
    """桌面机位：更靠画面下方、或框更矮的视为前面遮挡瓶。."""
    t1, c1, s1, x1, h1 = unpack_bottle(b1)
    t2, c2, s2, x2, h2 = unpack_bottle(b2)
    if x1 is not None and x2 is not None:
        y2_1, y2_2 = float(x1[3]), float(x2[3])
        if y2_1 > y2_2 + 10:
            return t1
        if y2_2 > y2_1 + 10:
            return t2
        return t1 if h1 <= h2 else t2
    if c1[1] > c2[1] + 8:
        return t1
    if c2[1] > c1[1] + 8:
        return t2
    return t1 if s1 <= s2 else t2


def occlusion_map(bottles: list) -> tuple[set[int], dict[int, int]]:
    """返回 (参与遮挡的id集合, {前面id: 后面id})."""
    close: set[int] = set()
    front_to_rear: dict[int, int] = {}
    n = len(bottles)
    for i in range(n):
        for j in range(i + 1, n):
            if not bottles_overlap(bottles[i], bottles[j]):
                continue
            t1 = int(bottles[i][0])
            t2 = int(bottles[j][0])
            close.add(t1)
            close.add(t2)
            f = int(front_tid(bottles[i], bottles[j]))
            r = t2 if f == t1 else t1
            front_to_rear[f] = r
    return close, front_to_rear


def crowded_bottle_tids(bottles: list) -> set[int]:
    close, _front = occlusion_map(bottles)
    return close


def too_short_for_label(rec, det_h: float) -> bool:
    rh = float(getattr(rec, "height", 0.0) or getattr(rec, "size", 0.0) or 0.0)
    if rh < 8 or det_h < 1:
        return False
    return det_h < 0.78 * rh


def height_mismatch(rec, det_h: float, ratio: float = 0.20) -> bool:
    rh = rec_height(rec)
    if rh < 8 or det_h < 1:
        return False
    return abs(rh - float(det_h)) / max(rh, float(det_h)) > ratio


def bottles_near(b1, b2, extra: float = 36.0) -> bool:
    if bottles_overlap(b1, b2):
        return True
    _t1, c1, s1, _x1, _h1 = unpack_bottle(b1)
    _t2, c2, s2, _x2, _h2 = unpack_bottle(b2)
    return float(np.linalg.norm(c1 - c2)) < 0.95 * 0.5 * (s1 + s2) + extra


def match_cost_box(curr_b, prev_b) -> float:
    _t1, c1, _s1, x1, h1 = unpack_bottle(curr_b)
    _t2, c2, _s2, x2, h2 = unpack_bottle(prev_b)
    d = float(np.linalg.norm(c1 - c2))
    d += 0.55 * abs(h1 - h2)
    if x1 is not None and x2 is not None:
        d += 80.0 * (1.0 - _xyxy_iou(x1, x2))
    return d


def snapshot_bottles(bottles: list) -> list:
    out = []
    for b in bottles:
        tid, center, size, xyxy, _h = unpack_bottle(b)
        out.append((tid, center.copy(), size, None if xyxy is None else xyxy.copy()))
    return out


def _park_record(ocr_cache: dict, parked: dict, rec, prefer_tid: int, frame_id: int, no_bind: set[int]) -> int:
    for k in list(ocr_cache):
        if ocr_cache[k] is rec:
            ocr_cache.pop(k, None)
    rec.parked = True
    rec.parked_frame = int(frame_id)
    rec.no_bind_tids = {int(t) for t in no_bind}
    pid = int(prefer_tid)
    while pid in parked:
        pid += 1
    parked[pid] = rec
    return pid


def _unpark_to(ocr_cache: dict, parked: dict, last_seen: dict, moved: list, pid: int, tid: int, frame_id: int) -> bool:
    rec = parked.pop(pid, None)
    if rec is None:
        return False
    rec.parked = False
    rec.no_bind_tids = set()
    ocr_cache[int(tid)] = rec
    last_seen[int(tid)] = int(frame_id)
    moved.append((int(pid), int(tid), str(getattr(rec, "reagent", ""))))
    return True


def ocr_blocked_tids(bind_mem: dict | None, frame_id: int) -> set[int]:
    until = (bind_mem or {}).get("skip_ocr_until") or {}
    return {int(t) for t, fr in until.items() if int(frame_id) <= int(fr)}


def expire_parked(parked: dict, frame_id: int, hold_frames: int) -> None:
    hold = max(8, int(hold_frames))
    for pid in list(parked):
        rec = parked[pid]
        age = int(frame_id) - int(getattr(rec, "parked_frame", frame_id))
        if age > hold:
            parked.pop(pid, None)


def follow_cost(rec, bottle, frame_id: int) -> float:
    """标签跟到当前框的代价：位置为主，框高只作弱约束（倾斜时高度会变）。."""
    _tid, center, _s, _x, h = unpack_bottle(bottle)
    pred = predicted_center(rec, frame_id)
    d = float(np.linalg.norm(center - pred))
    rh = rec_height(rec)
    if rh > 8 and h > 1:
        d += 0.25 * abs(rh - float(h))
    return d


def handle_pickup_loss(
    ocr_cache: dict,
    bottles: list,
    bind_mem: dict,
    last_seen: dict,
    frame_id: int,
    moved: list,
) -> set[int]:
    """仅两瓶重叠且一只真丢框、留下的明显是另一只时才停放。单瓶换 ID 直接跟着走。."""
    parked: dict = bind_mem.setdefault("parked", {})
    prev = bind_mem.get("prev") or []
    if len(prev) < 2 or not bottles:
        return set()

    curr_tids = {int(b[0]) for b in bottles}
    prev_tids = {int(b[0]) for b in prev}
    lost_tids = prev_tids - curr_tids
    if not lost_tids:
        return set()

    prev_map = {int(b[0]): b for b in prev}
    park_stayers: set[int] = set()

    for lt in list(lost_tids):
        rec = ocr_cache.get(lt)
        if rec is None or not getattr(rec, "reagent", ""):
            continue
        pb = prev_map.get(lt)
        if pb is None:
            continue
        overlapped = any(int(ob[0]) != lt and bottles_near(pb, ob) for ob in prev)

        scored: list[tuple[float, int]] = []
        for cb in bottles:
            tid, _c, _s, _x, h = unpack_bottle(cb)
            d = follow_cost(rec, cb, frame_id)
            # 留下的框明显更矮，不像同一只被拿起的瓶
            if overlapped and too_short_for_label(rec, h) and height_mismatch(rec, h, 0.32):
                continue
            scored.append((d, tid))
        scored.sort()
        limit = 2.2 * adaptive_max_dist(rec, 140.0)
        if scored and scored[0][0] <= limit and (len(scored) == 1 or scored[1][0] - scored[0][0] >= 18.0):
            _move_label(ocr_cache, last_seen, moved, lt, scored[0][1], frame_id)
            rec2 = ocr_cache.get(scored[0][1])
            if rec2 is not None:
                rec2.parked = False
            continue

        if not overlapped:
            continue
        near_now = [int(cb[0]) for cb in bottles if bottles_near(pb, cb)]
        if not near_now:
            continue
        _park_record(ocr_cache, parked, rec, lt, frame_id, set(near_now))
        park_stayers.update(near_now)

    return park_stayers


def try_unpark(
    ocr_cache: dict,
    bottles: list,
    bind_mem: dict,
    last_seen: dict,
    frame_id: int,
    moved: list,
    stayers: set[int],
    gap: float,
) -> None:
    parked: dict = bind_mem.setdefault("parked", {})
    if not parked or not bottles:
        return
    parsed = [unpack_bottle(b) for b in bottles]
    {tid: h for tid, _c, _s, _x, h in parsed}
    curr_tids = {tid for tid, _c, _s, _x, _h in parsed}

    def unlabeled(tid: int) -> bool:
        rec = ocr_cache.get(tid)
        return rec is None or not getattr(rec, "reagent", "")

    # 画面上只剩一只瓶：把停放标签还回去（移动换 ID 时最常见）
    if len(parsed) == 1:
        tid, _c, _s, _x, h = parsed[0]
        if unlabeled(tid):
            best_pid, best_d = None, 1e9
            for pid, rec in list(parked.items()):
                if not getattr(rec, "reagent", ""):
                    continue
                no_bind = set(getattr(rec, "no_bind_tids", ()) or ())
                if tid in no_bind and (too_short_for_label(rec, h) or height_mismatch(rec, h, 0.30)):
                    continue
                d = follow_cost(rec, bottles[0], frame_id)
                if d < best_d:
                    best_d, best_pid = d, pid
            if best_pid is not None:
                rec = parked[best_pid]
                if best_d <= 3.2 * adaptive_max_dist(rec, 140.0):
                    _unpark_to(ocr_cache, parked, last_seen, moved, best_pid, tid, frame_id)
                    return

    for pid in list(parked):
        rec = parked.get(pid)
        if rec is None:
            continue
        if pid in curr_tids and unlabeled(pid):
            _unpark_to(ocr_cache, parked, last_seen, moved, pid, pid, frame_id)

    parked = bind_mem.setdefault("parked", {})
    for pid in list(parked):
        rec = parked.get(pid)
        if rec is None or not getattr(rec, "reagent", ""):
            continue
        no_bind = set(getattr(rec, "no_bind_tids", ()) or ())
        cands: list[tuple[float, int]] = []
        for tid, center, _s, _x, h in parsed:
            if not unlabeled(tid):
                continue
            if tid in no_bind and (too_short_for_label(rec, h) or height_mismatch(rec, h, 0.30)):
                continue
            pred = predicted_center(rec, frame_id)
            d = float(np.linalg.norm(center - pred))
            d += 0.35 * abs(rec_height(rec) - float(h))
            cands.append((d, tid))
        cands.sort()
        if not cands:
            continue
        best_d, best_tid = cands[0]
        second = cands[1][0] if len(cands) > 1 else 1e9
        unique = (second - best_d) >= max(12.0, float(gap) * 0.5)
        limit = 2.6 * adaptive_max_dist(rec, 140.0)
        if best_d <= limit and (unique or len(cands) == 1):
            _unpark_to(ocr_cache, parked, last_seen, moved, pid, best_tid, frame_id)


def _move_label(ocr_cache: dict, last_seen: dict, moved: list, src: int, dst: int, frame_id: int) -> bool:
    if src == dst or src not in ocr_cache:
        return False
    dst_rec = ocr_cache.get(dst)
    if dst_rec is not None and getattr(dst_rec, "reagent", ""):
        return False
    rec = ocr_cache.pop(src)
    ocr_cache.pop(dst, None)
    ocr_cache[dst] = rec
    last_seen.pop(src, None)
    last_seen[dst] = frame_id
    moved.append((src, dst, str(getattr(rec, "reagent", ""))))
    return True


def _swap_labels(ocr_cache: dict, last_seen: dict, moved: list, a: int, b: int, frame_id: int) -> None:
    if a == b:
        return
    ra = ocr_cache.pop(a, None)
    rb = ocr_cache.pop(b, None)
    if ra is not None:
        ocr_cache[b] = ra
        last_seen[b] = frame_id
        if getattr(ra, "reagent", ""):
            moved.append((a, b, str(ra.reagent)))
    if rb is not None:
        ocr_cache[a] = rb
        last_seen[a] = frame_id
        if getattr(rb, "reagent", ""):
            moved.append((b, a, str(rb.reagent)))


def restore_occlusion_labels(
    ocr_cache: dict,
    bottles: list,
    det_xy: dict,
    det_h: dict,
    last_seen: dict,
    frame_id: int,
    moved: list,
) -> tuple[set[int], dict[int, int]]:
    """遮挡时：较矮/靠前的瓶不能占有高瓶已锁定标签；错绑则立刻搬回后瓶。."""
    crowded, front_to_rear = occlusion_map(bottles)
    for front, rear in list(front_to_rear.items()):
        if front not in det_h or rear not in det_h:
            continue
        fh, rh = float(det_h[front]), float(det_h[rear])
        frec = ocr_cache.get(front)
        rrec = ocr_cache.get(rear)
        f_lab = bool(frec is not None and getattr(frec, "reagent", ""))
        r_lab = bool(rrec is not None and getattr(rrec, "reagent", ""))

        def hcost(rec, h: float) -> float:
            if rec is None:
                return 0.0
            return abs(rec_height(rec) - float(h))

        if f_lab and r_lab:
            keep = hcost(frec, fh) + hcost(rrec, rh)
            swap = hcost(frec, rh) + hcost(rrec, fh)
            if too_short_for_label(frec, fh) or swap + 8.0 < keep:
                _swap_labels(ocr_cache, last_seen, moved, front, rear, frame_id)
                rec = ocr_cache.get(rear)
                if rec is not None and rear in det_xy:
                    rec.center = det_xy[rear].copy()
                    rec.frame_id = int(frame_id)
            continue

        if f_lab and not r_lab:
            tall_on_front = too_short_for_label(frec, fh) or (
                rh >= 1.12 * max(fh, 1.0) and rec_height(frec) > 1.08 * fh
            )
            if tall_on_front and _move_label(ocr_cache, last_seen, moved, front, rear, frame_id):
                rec = ocr_cache.get(rear)
                if rec is not None and rear in det_xy:
                    rec.center = det_xy[rear].copy()
                    rec.frame_id = int(frame_id)
            continue

        if r_lab and not f_lab:
            # 矮瓶自己的标签被跟到后瓶：还回去。后瓶已锁定的高瓶标签绝不给前瓶。
            own_short = rec_height(rrec) > 8 and rec_height(rrec) < 0.82 * rh
            closer_front = abs(rec_height(rrec) - fh) + 10.0 < abs(rec_height(rrec) - rh)
            if own_short and closer_front:
                _move_label(ocr_cache, last_seen, moved, rear, front, frame_id)
    return crowded, front_to_rear


def rebind_ocr_cache(
    ocr_cache: dict,
    bottles: list,
    frame_id: int,
    last_seen: dict,
    max_dist: float,
    gap: float,
    sticky: float,
    bind_mem: dict | None = None,
    hold_frames: int = 180,
) -> list[tuple[int, int, str]]:
    """把已锁定标签按预测位置重新绑到当前瓶子 track。返回 [(旧id, 新id, 标签)]."""
    moved: list[tuple[int, int, str]] = []
    if bind_mem is None:
        bind_mem = {}
    parked = bind_mem.setdefault("parked", {})
    expire_parked(parked, frame_id, hold_frames)

    parsed = [unpack_bottle(b) for b in bottles]
    det_xy = {tid: center for tid, center, _s, _x, _h in parsed}
    det_size = {tid: size for tid, _c, size, _x, _h in parsed}
    det_h = {tid: height for tid, _c, _s, _x, height in parsed}

    stayers = handle_pickup_loss(ocr_cache, bottles, bind_mem, last_seen, frame_id, moved)
    until = bind_mem.setdefault("skip_ocr_until", {})
    for t in stayers:
        until[int(t)] = max(int(until.get(t, 0)), int(frame_id) + 15)
    for t in list(until):
        if int(frame_id) > int(until[t]):
            until.pop(t, None)

    if not bottles:
        return moved

    crowded, _front_to_rear = restore_occlusion_labels(ocr_cache, bottles, det_xy, det_h, last_seen, frame_id, moved)
    front_tids = set(_front_to_rear.keys())

    locked = {tid: rec for tid, rec in ocr_cache.items() if getattr(rec, "reagent", "")}
    if not locked:
        for tid, center, size, _x, height in parsed:
            rec = ocr_cache.get(tid)
            if rec is not None:
                update_ocr_motion(rec, center, size, frame_id, height=height)
        try_unpark(ocr_cache, bottles, bind_mem, last_seen, frame_id, moved, stayers, gap)
        bind_mem["prev"] = snapshot_bottles(bottles)
        return moved

    def dist_rec_to_tid(rec, tid: int) -> float:
        pred = predicted_center(rec, frame_id)
        d = float(np.linalg.norm(det_xy[tid] - pred))
        rs = float(getattr(rec, "size", 0.0) or 0.0)
        if rs > 1 and det_size[tid] > 1:
            d += 0.2 * abs(rs - det_size[tid])
        rh = rec_height(rec)
        if rh > 1 and det_h[tid] > 1:
            d += 0.55 * abs(rh - det_h[tid])
        return d

    def nearest_dets(rec, exclude: set[int] | None = None) -> list[tuple[float, int]]:
        rows = []
        for tid in det_xy:
            if exclude and tid in exclude:
                continue
            rows.append((dist_rec_to_tid(rec, tid), tid))
        rows.sort()
        return rows

    keep: set[int] = set()
    freeze: set[int] = set()
    free_old: list[tuple[int, object]] = []

    for oid, rec in list(locked.items()):
        limit = adaptive_max_dist(rec, max_dist)
        jump_lim = max(48.0, 0.42 * float(getattr(rec, "size", 0.0) or 0.0))
        if oid in det_xy:
            if too_short_for_label(rec, det_h[oid]) and oid not in crowded and len(parsed) > 1:
                free_old.append((oid, rec))
                continue
            d0 = dist_rec_to_tid(rec, oid)
            near = nearest_dets(rec)
            if not near:
                keep.add(oid)
                freeze.add(oid)
                continue
            best_d, best_tid = near[0]
            second = near[1][0] if len(near) > 1 else 1e9
            unique = (second - best_d) >= gap
            jumped = d0 > jump_lim
            if oid in crowded or best_tid in crowded:
                keep.add(oid)
                freeze.add(oid)
                continue
            if best_tid == oid and not jumped:
                keep.add(oid)
            elif jumped and unique and best_d <= limit:
                free_old.append((oid, rec))
            elif (not jumped) and d0 <= sticky and d0 <= best_d + 8.0:
                keep.add(oid)
            elif unique and best_d <= limit:
                free_old.append((oid, rec))
            else:
                keep.add(oid)
        else:
            if oid in crowded:
                freeze.add(oid)
                keep.add(oid)
            else:
                free_old.append((oid, rec))

    taken_new = set(keep)
    taken_old = set(keep)
    claims: list[tuple[float, int, int]] = []
    for oid, rec in free_old:
        limit = adaptive_max_dist(rec, max_dist)
        ranked = nearest_dets(rec, exclude=taken_new)
        if not ranked:
            continue
        best_d, best_tid = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 1e9
        if best_d > limit or best_tid in crowded:
            continue
        if best_tid in stayers and len(parsed) > 1:
            continue
        if best_tid in front_tids and too_short_for_label(rec, det_h[best_tid]):
            continue
        unique = second - best_d >= gap
        contended = False
        for oid2, rec2 in free_old:
            if oid2 == oid:
                continue
            ranked2 = nearest_dets(rec2, exclude=taken_new)
            if ranked2 and ranked2[0][1] == best_tid and ranked2[0][0] <= best_d:
                contended = True
                break
        if unique or (best_d <= sticky and not contended):
            claims.append((best_d, best_tid, oid))

    claims.sort()
    for _d, tid, oid in claims:
        if tid in taken_new or oid in taken_old:
            continue
        taken_new.add(tid)
        taken_old.add(oid)
        rec = ocr_cache.pop(oid)
        ocr_cache[tid] = rec
        last_seen.pop(oid, None)
        last_seen[tid] = frame_id
        moved.append((oid, tid, str(rec.reagent)))

    leftover_old = [(oid, rec) for oid, rec in free_old if oid not in taken_old and oid in ocr_cache]
    leftover_new = [tid for tid, _c, _s, _x, _h in parsed if tid not in taken_new]
    if len(leftover_old) == 1 and len(leftover_new) == 1:
        oid, rec = leftover_old[0]
        tid = leftover_new[0]
        steal_ok = (tid not in stayers) or len(parsed) == 1
        if steal_ok and (tid not in crowded or len(parsed) == 1):
            scale = 3.2 if len(bottles) == 1 or len(locked) == 1 else 1.15
            limit = scale * adaptive_max_dist(rec, max_dist)
            if dist_rec_to_tid(rec, tid) <= limit:
                rec = ocr_cache.pop(oid)
                ocr_cache[tid] = rec
                last_seen.pop(oid, None)
                last_seen[tid] = frame_id
                moved.append((oid, tid, str(rec.reagent)))
                taken_new.add(tid)

    for tid, center, size, _x, height in parsed:
        rec = ocr_cache.get(tid)
        if rec is None:
            continue
        if tid in crowded or tid in freeze:
            rec.frame_id = int(frame_id)
            continue
        update_ocr_motion(rec, center, size, frame_id, height=height)
    try_unpark(ocr_cache, bottles, bind_mem, last_seen, frame_id, moved, stayers, gap)
    bind_mem["prev"] = snapshot_bottles(bottles)
    return moved


def assign_spatial_tids(
    detections: list[tuple[np.ndarray, float]],
    ocr_cache: dict,
    frame_id: int,
    max_dist: float,
    gap: float,
    next_tid: int,
    sticky: float = 110.0,
) -> tuple[list[int], int]:
    """无 ByteTrack 时，按预测位置给瓶子分配稳定 id，避免网格跳动丢缓存。."""
    n = len(detections)
    tids = [0] * n
    used_old: set[int] = set()
    claims: list[tuple[float, int, int]] = []
    det_xy = [np.asarray(c, dtype=np.float32).reshape(-1)[:2] for c, _s in detections]

    def dist_oid(oid: int, i: int) -> float:
        rec = ocr_cache[oid]
        pred = predicted_center(rec, frame_id)
        d = float(np.linalg.norm(det_xy[i] - pred))
        rs = float(getattr(rec, "size", 0.0) or 0.0)
        size = float(detections[i][1])
        if rs > 1 and size > 1:
            d += 0.2 * abs(rs - size)
        return d

    for i in range(n):
        ranked = sorted((dist_oid(oid, i), int(oid)) for oid in ocr_cache)
        if not ranked:
            continue
        best_d, oid = ranked[0]
        second = ranked[1][0] if len(ranked) > 1 else 1e9
        limit = adaptive_max_dist(ocr_cache[oid], max_dist)
        if best_d > limit:
            continue
        unique = second - best_d >= gap
        contended = False
        for j in range(n):
            if j == i:
                continue
            other = sorted((dist_oid(oid2, j), int(oid2)) for oid2 in ocr_cache)
            if other and other[0][1] == oid and other[0][0] <= best_d:
                contended = True
                break
        if unique or (best_d <= sticky and not contended):
            claims.append((best_d, i, oid))

    claims.sort()
    taken_det: set[int] = set()
    for _d, i, oid in claims:
        if i in taken_det or oid in used_old:
            continue
        taken_det.add(i)
        used_old.add(oid)
        tids[i] = oid

    leftover = [i for i in range(n) if i not in taken_det]
    leftover_old = [oid for oid in ocr_cache if oid not in used_old]
    if len(leftover) == 1 and len(leftover_old) == 1:
        i = leftover[0]
        oid = leftover_old[0]
        center = np.asarray(detections[i][0], dtype=np.float32).reshape(-1)[:2]
        rec = ocr_cache[oid]
        d = float(np.linalg.norm(center - predicted_center(rec, frame_id)))
        if d <= (2.4 if n == 1 else 1.05) * adaptive_max_dist(rec, max_dist):
            tids[i] = oid
            leftover = []

    for i in leftover:
        while next_tid in ocr_cache or next_tid in tids:
            next_tid += 1
        tids[i] = next_tid
        next_tid += 1
    return tids, next_tid
