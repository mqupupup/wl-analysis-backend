"""
error_detection_engine.py (v2)

修正：
  - bounce: 多信号联合判断
  - elbow_flare: 无肩外展数据 → INSUFFICIENT_DATA
  - hip_lift: 无髋部数据 → INSUFFICIENT_DATA
  - Set 级别聚合
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import numpy as np
import warnings

from app.domain.models import RepContext
from app.domain.enums import ValidationStatus, ErrorStatus, ErrorSeverity


@dataclass
class ErrorDetection:
    error_id: str
    rep_index: int
    status: ErrorStatus
    severity: Optional[ErrorSeverity] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    confidence: float = 0.0
    detail: str = ""


@dataclass
class SetLevelError:
    error_id: str
    display_name: str
    occurrences: List[int] = field(default_factory=list)
    frequency: float = 0.0
    worst_severity: Optional[ErrorSeverity] = None
    avg_value: Optional[float] = None


# ═══════════════════════════════════════
#  规则函数
# ═══════════════════════════════════════

def detect_incomplete_rom(rep: RepContext, min_rom: float = 60.0) -> ErrorDetection:
    eid = "bench_incomplete_rom"
    if rep.actual_rom <= 0:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="ROM 不可用")
    if rep.actual_rom < min_rom:
        sev = ErrorSeverity.SEVERE if rep.actual_rom < min_rom * 0.6 else ErrorSeverity.MODERATE
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED, sev,
                              value=rep.actual_rom, threshold=min_rom, confidence=0.9,
                              detail=f"actual_rom={rep.actual_rom:.1f}° < {min_rom}°")
    return ErrorDetection(eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
                          value=rep.actual_rom, threshold=min_rom, confidence=0.9)


def detect_incomplete_lockout(rep: RepContext, threshold: float = 160.0) -> ErrorDetection:
    eid = "bench_incomplete_lockout"
    if rep.top2_angle <= 0:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA)
    if rep.top2_angle < threshold:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED,
                              ErrorSeverity.MODERATE, value=rep.top2_angle,
                              threshold=threshold, confidence=0.8,
                              detail=f"TOP2 angle={rep.top2_angle:.1f}°")
    return ErrorDetection(eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
                          value=rep.top2_angle, threshold=threshold, confidence=0.8)


def detect_elbow_flare(rep: RepContext) -> ErrorDetection:
    """需要肩外展角数据，当前不可用"""
    return ErrorDetection(
        "bench_elbow_flare", rep.rep_index,
        ErrorStatus.INSUFFICIENT_DATA,
        detail="需要肩外展角数据，当前不可用",
        confidence=0.0,
    )


def detect_bounce(rep: RepContext) -> ErrorDetection:
    """砸胸检测 — 多信号联合判断"""
    eid = "bench_bounce"

    if rep.pre_bottom_velocity is None:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="无 bottom 前速度数据")

    PRE_VEL_THRESH = 180.0
    DWELL_THRESH = 0.08
    REVERSAL_THRESH = 2
    ACCEL_THRESH = 3000.0

    signals_hit = 0
    details = []

    fast_descent = rep.pre_bottom_velocity > PRE_VEL_THRESH
    if fast_descent:
        signals_hit += 1
        details.append(f"pre_vel={rep.pre_bottom_velocity:.0f}°/s")

    short_dwell = rep.bottom_dwell_time < DWELL_THRESH
    if short_dwell:
        signals_hit += 1
        details.append(f"dwell={rep.bottom_dwell_time * 1000:.0f}ms")

    fast_reversal = rep.direction_reversal_frames <= REVERSAL_THRESH
    if fast_reversal:
        signals_hit += 1
        details.append(f"reversal={rep.direction_reversal_frames}frames")

    high_accel = (rep.bottom_acceleration is not None
                  and abs(rep.bottom_acceleration) > ACCEL_THRESH)
    if high_accel:
        signals_hit += 1
        details.append(f"accel={rep.bottom_acceleration:.0f}°/s²")

    if signals_hit >= 3:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED,
                              ErrorSeverity.SEVERE, value=rep.pre_bottom_velocity,
                              threshold=PRE_VEL_THRESH, confidence=0.9,
                              detail="; ".join(details))
    elif signals_hit == 2 and (fast_descent and (short_dwell or fast_reversal)):
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED,
                              ErrorSeverity.MODERATE, value=rep.pre_bottom_velocity,
                              threshold=PRE_VEL_THRESH, confidence=0.75,
                              detail="; ".join(details))
    else:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
                              value=rep.pre_bottom_velocity, confidence=0.7,
                              detail="; ".join(details) if details else "正常触胸")


def detect_hip_lift(rep: RepContext) -> ErrorDetection:
    """需要髋部角度数据，当前不可用。注意：应检测臀部离凳(butt-off)，而非单纯起桥(arch)。"""
    return ErrorDetection(
        "bench_butt_off_bench", rep.rep_index,
        ErrorStatus.INSUFFICIENT_DATA,
        detail="需要髋部/臀部接触数据，当前不可用",
        confidence=0.0,
    )


def detect_asymmetric_push(rep: RepContext, max_asym: float = 12.0) -> ErrorDetection:
    eid = "bench_asymmetric_push"

    # 关键修复：双侧有效比例不足时，绝对不能判定"左右不对称"
    # 这是"测不到"，不是"做得差"
    bilateral = float(getattr(rep, "bilateral_valid_ratio", 0.0))
    if bilateral < 0.65:
        return ErrorDetection(
            eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
            detail=f"双侧有效比例不足: {bilateral:.2f} (<0.65)，跳过对称性判断",
            confidence=0.0,
        )

    if rep.left_elbow is None or rep.right_elbow is None:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="缺少单侧数据")

    cs = max(0, rep.concentric_start - rep.start_frame)
    ce = min(len(rep.left_elbow), rep.concentric_end - rep.start_frame)
    if ce - cs < 5:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="concentric 有效帧不足")

    seg_l = np.asarray(rep.left_elbow[cs: ce], dtype=float)
    seg_r = np.asarray(rep.right_elbow[cs: ce], dtype=float)
    valid = np.isfinite(seg_l) & np.isfinite(seg_r)

    if valid.sum() < 5:
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.INSUFFICIENT_DATA,
                              detail="双侧同时有效帧不足")

    # 用 median 替代 mean，抗异常帧
    asym = float(np.nanmedian(np.abs(seg_l[valid] - seg_r[valid])))

    if asym > max_asym:
        sev = ErrorSeverity.SEVERE if asym > max_asym * 2 else ErrorSeverity.MODERATE
        return ErrorDetection(eid, rep.rep_index, ErrorStatus.DETECTED, sev,
                              value=asym, threshold=max_asym, confidence=0.8,
                              detail=f"asym={asym:.1f}°")
    return ErrorDetection(eid, rep.rep_index, ErrorStatus.NOT_DETECTED,
                          value=asym, threshold=max_asym, confidence=0.8)


BENCH_RULES = [
    detect_incomplete_rom,
    detect_incomplete_lockout,
    detect_elbow_flare,
    detect_bounce,
    detect_hip_lift,
    detect_asymmetric_push,
]


class ErrorDetectionEngineFixed:

    def detect_set(self, reps: List[RepContext]) -> List[SetLevelError]:
        all_det: Dict[str, List[ErrorDetection]] = {}

        for rep in reps:
            if rep.validation_status != ValidationStatus.VALID:
                continue
            for fn in BENCH_RULES:
                d = fn(rep)
                all_det.setdefault(d.error_id, []).append(d)

        valid_count = sum(1 for r in reps if r.validation_status == ValidationStatus.VALID)
        results: List[SetLevelError] = []

        for eid, dets in all_det.items():
            hits = [d for d in dets if d.status == ErrorStatus.DETECTED]
            if not hits:
                continue
            vals = [d.value for d in hits if d.value is not None]
            sevs = [d.severity for d in hits if d.severity]
            worst = max(sevs, key=lambda s: list(ErrorSeverity).index(s)) if sevs else None

            results.append(SetLevelError(
                error_id=eid,
                display_name=self._name(eid),
                occurrences=[d.rep_index for d in hits],
                frequency=len(hits) / max(1, valid_count),
                worst_severity=worst,
                avg_value=float(np.mean(vals)) if vals else None,
            ))

        return results

    @staticmethod
    def _name(eid: str) -> str:
        return {
            "bench_incomplete_rom": "动作幅度不足",
            "bench_incomplete_lockout": "锁定不完全",
            "bench_elbow_flare": "肘部外展",
            "bench_bounce": "砸胸/弹胸",
            "bench_butt_off_bench": "臀部离凳",
            "bench_asymmetric_push": "左右发力不对称",
        }.get(eid, eid)